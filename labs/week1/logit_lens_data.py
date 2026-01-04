"""
Efficient logit lens data collection for NDIF remote execution.

This module provides functions to collect logit lens data from transformer
language models, optimized for remote execution via NDIF where bandwidth
between server and client is the primary bottleneck.

Key insight: For a 70B model with 80 layers and 128k vocabulary, naive
transmission of full logits would require ~40GB per prompt. By computing
top-k and trajectory extraction on the server, we reduce this to ~1MB.

Two collection modes:
1. top_k_per_layer: Get top-k predictions at each layer/position
   - Returns: top_indices[n_layers, seq_len, k], top_probs[n_layers, seq_len, k]
   - Bandwidth: O(n_layers * seq_len * k) - typically ~100KB

2. track_across_layers: Identify tokens appearing in top-k at ANY layer,
   then track their probabilities across ALL layers
   - Enables trajectory visualization showing how predictions evolve
   - Returns: tracked_indices per position, tracked_probs[n_layers, n_tracked]
   - Bandwidth: O(n_layers * seq_len * n_tracked) where n_tracked << vocab_size

Model Support:
    Supports multiple transformer architectures via a registry pattern.
    Each model family specifies paths to layers, norm, and lm_head.
    Registry entries can be strings (dot-separated paths) or callables
    for complex/custom access patterns.

Functions:
    collect_logit_lens_topk: Simple version, downloads full logits (inefficient)
    collect_logit_lens_topk_efficient: Server-side reduction (recommended)
    decode_tracked_tokens: Convert token indices to strings
    get_model_config: Get or auto-detect model configuration

Example:
    >>> from nnsight import LanguageModel
    >>> model = LanguageModel("meta-llama/Llama-3.1-70B", device_map="auto")
    >>> data = collect_logit_lens_topk_efficient(
    ...     "The quick brown fox",
    ...     model,
    ...     top_k=5,
    ...     track_across_layers=True,
    ...     remote=True
    ... )
    >>> # data ready for LogitLensWidget visualization
"""

import torch
from typing import List, Dict, Tuple, Optional, Union, Callable, Any


# =============================================================================
# Model Configuration Registry
# =============================================================================
#
# Each entry maps a model type to its architecture-specific accessors.
# Values can be:
#   - String: Dot-separated path (e.g., "model.layers")
#   - Callable: Function taking model (and optionally hidden state)
#
# Required keys:
#   - layers: Path to layer list/ModuleList
#   - norm: Final layer norm (module or callable(model, hidden) -> normalized)
#   - lm_head: Language model head (module or weight matrix)
#   - n_layers: Number of layers (string path to config attr, or callable)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama": {
        "layers": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
        "n_layers": "config.num_hidden_layers",
    },
    "mistral": {
        "layers": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
        "n_layers": "config.num_hidden_layers",
    },
    "qwen2": {
        "layers": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
        "n_layers": "config.num_hidden_layers",
    },
    "gpt2": {
        "layers": "transformer.h",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
        "n_layers": "config.n_layer",
    },
    "gptj": {
        "layers": "transformer.h",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
        "n_layers": "config.n_layer",
    },
    "gpt_neox": {
        "layers": "gpt_neox.layers",
        "norm": "gpt_neox.final_layer_norm",
        "lm_head": "embed_out",
        "n_layers": "config.num_hidden_layers",
    },
    "olmo": {
        "layers": "model.transformer.blocks",
        "norm": "model.transformer.ln_f",
        "lm_head": "model.transformer.ff_out",
        "n_layers": "config.n_layers",
    },
    "phi": {
        "layers": "model.layers",
        "norm": "model.final_layernorm",
        "lm_head": "lm_head",
        "n_layers": "config.num_hidden_layers",
    },
    "gemma": {
        "layers": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
        "n_layers": "config.num_hidden_layers",
    },
}

# Aliases for common model names
MODEL_ALIASES = {
    "llama2": "llama",
    "llama3": "llama",
    "codellama": "llama",
    "pythia": "gpt_neox",
    "gpt-j": "gptj",
    "gpt-neox": "gpt_neox",
    "qwen": "qwen2",
    "gemma2": "gemma",
    "phi3": "phi",
    "phi-3": "phi",
}


def resolve_accessor(model, accessor: Union[str, Callable]) -> Any:
    """
    Resolve an accessor to get a module, value, or callable result.

    Args:
        model: The nnsight LanguageModel
        accessor: Either a dot-separated path string or a callable

    Returns:
        The resolved module, attribute, or callable result

    Examples:
        >>> resolve_accessor(model, "model.layers")  # Returns layers ModuleList
        >>> resolve_accessor(model, "config.num_hidden_layers")  # Returns int
        >>> resolve_accessor(model, lambda m: m.custom.path)  # Callable
    """
    if callable(accessor):
        return accessor(model)

    # String path traversal
    obj = model
    for attr in accessor.split("."):
        obj = getattr(obj, attr)
    return obj


def apply_module_or_callable(model, accessor: Union[str, Callable], hidden):
    """
    Apply a norm or lm_head accessor to hidden states.

    Handles three cases:
    1. String path to a module -> resolve and call module(hidden)
    2. Callable(model) returning a module -> call module(hidden)
    3. Callable(model, hidden) -> call directly with hidden
    4. Callable(model) returning weight matrix -> hidden @ weights

    Args:
        model: The nnsight LanguageModel
        accessor: String path or callable
        hidden: Hidden state tensor to process

    Returns:
        Processed tensor (normalized or logits)
    """
    if callable(accessor):
        # Check if it's a callable that takes hidden directly
        import inspect
        sig = inspect.signature(accessor)
        if len(sig.parameters) >= 2:
            # Callable(model, hidden) -> direct application
            return accessor(model, hidden)
        else:
            # Callable(model) -> returns module or weights
            resolved = accessor(model)
    else:
        # String path -> resolve to module
        resolved = resolve_accessor(model, accessor)

    # Now apply the resolved object
    if hasattr(resolved, 'forward') or hasattr(resolved, '__call__'):
        # It's a module, call it
        return resolved(hidden)
    else:
        # Assume it's a weight matrix (for tied embeddings)
        return hidden @ resolved


def detect_model_type(model) -> str:
    """
    Auto-detect model type from config.

    Args:
        model: nnsight LanguageModel

    Returns:
        Model type string (key in MODEL_CONFIGS)

    Raises:
        ValueError: If model type cannot be detected
    """
    # Try model_type from config
    model_type = getattr(model.config, "model_type", "").lower()

    # Check direct match
    if model_type in MODEL_CONFIGS:
        return model_type

    # Check aliases
    if model_type in MODEL_ALIASES:
        return MODEL_ALIASES[model_type]

    # Try architectures field
    archs = getattr(model.config, "architectures", [])
    for arch in archs:
        arch_lower = arch.lower()
        for key in MODEL_CONFIGS:
            if key in arch_lower:
                return key
        for alias, target in MODEL_ALIASES.items():
            if alias.replace("-", "").replace("_", "") in arch_lower:
                return target

    raise ValueError(
        f"Unknown model type: {model_type}. "
        f"Supported types: {list(MODEL_CONFIGS.keys())}. "
        f"You can pass model_type explicitly or add a config to MODEL_CONFIGS."
    )


def get_model_config(model, model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model configuration, auto-detecting if not specified.

    Args:
        model: nnsight LanguageModel
        model_type: Explicit model type, or None to auto-detect

    Returns:
        Configuration dict with layers, norm, lm_head, n_layers accessors
    """
    if model_type is None:
        model_type = detect_model_type(model)

    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    return MODEL_CONFIGS[model_type]


def get_value(saved):
    """Helper to get value from saved tensor."""
    try:
        return saved.value
    except AttributeError:
        return saved


def collect_logit_lens_topk(
    prompt: str,
    model,
    top_k: int = 5,
    track_across_layers: bool = False,
    remote: bool = True,
    layers: Optional[List[int]] = None,
    model_type: Optional[str] = None,
) -> Dict:
    """
    Collect logit lens data efficiently for visualization.

    This version downloads full logits from the server before processing.
    For bandwidth-efficient collection, use collect_logit_lens_topk_efficient.

    Args:
        prompt: Input text
        model: nnsight LanguageModel
        top_k: Number of top predictions to track
        track_across_layers: If True, find tokens in top-k at ANY layer,
            then track their probabilities at ALL layers. If False, just
            return top-k at each layer independently.
        remote: Use NDIF remote execution
        layers: Specific layers to analyze (default: all layers)
        model_type: Model architecture type (auto-detected if None)

    Returns:
        Dict with:
            - tokens: List of input token strings
            - layers: List of layer indices analyzed
            - If track_across_layers=False:
                - top_indices: [n_layers, seq_len, k] token indices
                - top_probs: [n_layers, seq_len, k] probabilities
            - If track_across_layers=True:
                - tracked_indices: [seq_len, n_tracked] unique token indices per position
                - tracked_probs: [n_layers, seq_len, n_tracked] probabilities
                - top_indices: [n_layers, seq_len, k] which of tracked are top-k per layer
    """
    # Get model-specific configuration
    cfg = get_model_config(model, model_type)
    n_layers = resolve_accessor(model, cfg["n_layers"])
    model_layers = resolve_accessor(model, cfg["layers"])

    if layers is None:
        layers = list(range(n_layers))

    # Tokenize
    token_ids = model.tokenizer.encode(prompt)
    token_strs = [model.tokenizer.decode([t]) for t in token_ids]
    seq_len = len(token_ids)

    # Collect logits at all layers - use list.save()
    saved_logits = None
    with model.trace(prompt, remote=remote):
        logits_list = []
        for layer_idx in layers:
            hidden = model_layers[layer_idx].output[0]
            normed = apply_module_or_callable(model, cfg["norm"], hidden)
            logits = apply_module_or_callable(model, cfg["lm_head"], normed)
            # Remove batch dim if present: [seq_len, vocab]
            seq_logits = logits[0] if len(logits.shape) == 3 else logits
            logits_list.append(seq_logits)
        saved_logits = logits_list.save()

    # Stack into [n_layers, seq_len, vocab]
    all_logits = torch.stack([get_value(t).float() for t in saved_logits])
    n_analyzed = len(layers)

    # Compute probabilities
    all_probs = torch.softmax(all_logits, dim=-1)

    # Get top-k at each layer/position
    top_probs, top_indices = all_probs.topk(top_k, dim=-1)
    # Shape: [n_layers, seq_len, k]

    if not track_across_layers:
        return {
            "tokens": token_strs,
            "layers": layers,
            "top_indices": top_indices,
            "top_probs": top_probs,
        }

    # Track across layers mode:
    # Find union of top-k tokens at each position across all layers
    tracked_indices_per_pos = []
    tracked_probs_per_pos = []

    for pos in range(seq_len):
        # Collect all tokens that appear in top-k at any layer for this position
        all_top_at_pos = top_indices[:, pos, :].reshape(-1)  # [n_layers * k]
        unique_tokens = torch.unique(all_top_at_pos)

        # Get probabilities for these tokens at all layers
        # all_probs[:, pos, :] is [n_layers, vocab]
        probs_for_tracked = all_probs[:, pos, :][:, unique_tokens]  # [n_layers, n_unique]

        tracked_indices_per_pos.append(unique_tokens)
        tracked_probs_per_pos.append(probs_for_tracked)

    return {
        "tokens": token_strs,
        "layers": layers,
        "tracked_indices": tracked_indices_per_pos,  # List of [n_tracked] per position
        "tracked_probs": tracked_probs_per_pos,  # List of [n_layers, n_tracked] per position
        "top_indices": top_indices,  # [n_layers, seq_len, k] for identifying top-k
        "top_probs": top_probs,  # [n_layers, seq_len, k]
    }


def collect_logit_lens_topk_efficient(
    prompt: str,
    model,
    top_k: int = 5,
    track_across_layers: bool = False,
    remote: bool = True,
    layers: Optional[List[int]] = None,
    model_type: Optional[str] = None,
) -> Dict:
    """
    Collect logit lens data with server-side reduction for minimal bandwidth.

    This is the recommended function for NDIF remote execution. All heavy
    computation (softmax, top-k, unique, indexing) happens on the server,
    transmitting only the essential results back to the client.

    Bandwidth comparison for Llama-70B (80 layers, 128k vocab, 20 tokens):
        - Naive (full logits):     80 * 20 * 128k * 4 bytes = 819 MB
        - This function (top-5):   80 * 20 * 5 * 8 bytes   = 64 KB
        - With trajectories:       + ~80 * 20 * 50 * 4     = ~320 KB total

    The trajectory tracking identifies which tokens appear in top-k at ANY
    layer for each position, then extracts just those tokens' probabilities
    across all layers. This enables visualization of how predictions evolve
    without downloading the full probability distribution.

    Args:
        prompt: Input text to analyze
        model: nnsight LanguageModel instance
        top_k: Number of top predictions per layer/position (default: 5)
        track_across_layers: If True, track probability trajectories for
            tokens appearing in top-k at any layer. Required for trajectory
            visualization in LogitLensWidget. (default: False)
        remote: Use NDIF remote execution (default: True)
        layers: Specific layer indices to analyze (default: all layers)
        model_type: Model architecture type (auto-detected if None).
            Supported: llama, gpt2, gptj, gpt_neox, pythia, olmo, phi, gemma, etc.

    Returns:
        Dict containing:
            tokens: List[str] - Input token strings
            layers: List[int] - Layer indices analyzed
            top_indices: Tensor[n_layers, seq_len, k] - Top-k token indices
            top_probs: Tensor[n_layers, seq_len, k] - Top-k probabilities

        If track_across_layers=True, additionally:
            tracked_indices: List[Tensor] - Unique token indices per position
            tracked_probs: List[Tensor[n_layers, n_tracked]] - Trajectories

    Example:
        >>> data = collect_logit_lens_topk_efficient(
        ...     "The capital of France is",
        ...     model,
        ...     top_k=5,
        ...     track_across_layers=True,
        ...     remote=True
        ... )
        >>> # Check top prediction at final layer for last token
        >>> final_layer = len(data["layers"]) - 1
        >>> last_pos = len(data["tokens"]) - 1
        >>> top_token_idx = data["top_indices"][final_layer, last_pos, 0]
        >>> print(model.tokenizer.decode([top_token_idx]))  # " Paris"
    """
    # Get model-specific configuration
    cfg = get_model_config(model, model_type)
    n_layers = resolve_accessor(model, cfg["n_layers"])
    model_layers = resolve_accessor(model, cfg["layers"])

    if layers is None:
        layers = list(range(n_layers))

    # Tokenize
    token_ids = model.tokenizer.encode(prompt)
    token_strs = [model.tokenizer.decode([t]) for t in token_ids]

    if not track_across_layers:
        # Simple mode: just get top-k at each layer
        saved_data = None
        with model.trace(prompt, remote=remote):
            results = []
            for layer_idx in layers:
                hidden = model_layers[layer_idx].output[0]
                normed = apply_module_or_callable(model, cfg["norm"], hidden)
                logits = apply_module_or_callable(model, cfg["lm_head"], normed)
                seq_logits = logits[0] if len(logits.shape) == 3 else logits
                probs = torch.softmax(seq_logits, dim=-1)
                top_probs, top_indices = probs.topk(top_k, dim=-1)
                # Save as tuple
                results.append((top_probs, top_indices))
            saved_data = results.save()

        # Unpack results
        top_probs_list = []
        top_indices_list = []
        for probs, indices in saved_data:
            top_probs_list.append(get_value(probs).float())
            top_indices_list.append(get_value(indices))

        return {
            "tokens": token_strs,
            "layers": layers,
            "top_indices": torch.stack(top_indices_list),
            "top_probs": torch.stack(top_probs_list),
        }

    # Track across layers mode - single-pass approach
    # All computation happens on the server: top-k, unique, and prob extraction

    seq_len = len(token_ids)
    num_layers = len(layers)

    saved_results = None
    with model.trace(prompt, remote=remote):
        # First collect all probs and top-k for each layer
        all_probs = []  # [n_layers, seq_len, vocab]
        all_top_indices = []  # [n_layers, seq_len, k]
        all_top_probs = []  # [n_layers, seq_len, k]

        for layer_idx in layers:
            hidden = model_layers[layer_idx].output[0]
            normed = apply_module_or_callable(model, cfg["norm"], hidden)
            logits = apply_module_or_callable(model, cfg["lm_head"], normed)
            seq_logits = logits[0] if len(logits.shape) == 3 else logits
            probs = torch.softmax(seq_logits, dim=-1)
            top_p, top_i = probs.topk(top_k, dim=-1)
            all_probs.append(probs)
            all_top_indices.append(top_i)
            all_top_probs.append(top_p)

        # Stack top-k results: [n_layers, seq_len, k]
        stacked_top_indices = torch.stack(all_top_indices)
        stacked_top_probs = torch.stack(all_top_probs)

        # For each position, find unique tokens and extract their probs
        tracked_results = []
        for pos in range(seq_len):
            # Get all top-k indices at this position across layers: [n_layers * k]
            pos_indices = stacked_top_indices[:, pos, :].reshape(-1)
            unique_tokens = torch.unique(pos_indices)

            # Extract probs for these tokens at all layers
            pos_tracked_probs = []
            for li in range(num_layers):
                probs_at_layer = all_probs[li][pos, unique_tokens]  # [n_unique]
                pos_tracked_probs.append(probs_at_layer)

            # Stack: [n_layers, n_unique]
            tracked_results.append((unique_tokens, torch.stack(pos_tracked_probs)))

        # Save everything
        saved_results = (stacked_top_indices, stacked_top_probs, tracked_results).save()

    # Unpack results
    top_indices, top_probs, tracked_list = saved_results
    top_indices = get_value(top_indices)
    top_probs = get_value(top_probs).float()

    tracked_indices_per_pos = []
    tracked_probs_per_pos = []
    for unique_tokens, probs_matrix in tracked_list:
        tracked_indices_per_pos.append(get_value(unique_tokens))
        tracked_probs_per_pos.append(get_value(probs_matrix).float())

    return {
        "tokens": token_strs,
        "layers": layers,
        "tracked_indices": tracked_indices_per_pos,
        "tracked_probs": tracked_probs_per_pos,
        "top_indices": top_indices,
        "top_probs": top_probs,
    }


def decode_tracked_tokens(data: Dict, tokenizer) -> Dict[int, List[str]]:
    """
    Decode tracked token indices to strings.

    Returns: {position: [token_str, ...]}
    """
    result = {}
    for pos, indices in enumerate(data["tracked_indices"]):
        result[pos] = [tokenizer.decode([idx.item()]) for idx in indices]
    return result
