"""
Efficient logit lens data collection for NDIF remote execution.

Provides two modes:
1. top_k_per_layer: Get top-k predictions at each layer/position
2. track_across_layers: Identify tokens in top-k at ANY layer, then track
   their probabilities across ALL layers (enables trajectory visualization)
"""

import torch
from typing import List, Dict, Tuple, Optional, Union


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
) -> Dict:
    """
    Collect logit lens data efficiently for visualization.

    Args:
        prompt: Input text
        model: nnsight LanguageModel
        top_k: Number of top predictions to track
        track_across_layers: If True, find tokens in top-k at ANY layer,
            then track their probabilities at ALL layers. If False, just
            return top-k at each layer independently.
        remote: Use NDIF remote execution
        layers: Specific layers to analyze (default: all layers)

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
    n_layers = model.config.num_hidden_layers
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
            hidden = model.model.layers[layer_idx].output[0]
            logits = model.lm_head(model.model.norm(hidden))
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
) -> Dict:
    """
    More efficient version that does top-k extraction on the server.

    This transmits only the necessary data back from NDIF, not full logits.
    """
    n_layers = model.config.num_hidden_layers
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
                hidden = model.model.layers[layer_idx].output[0]
                logits = model.lm_head(model.model.norm(hidden))
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
    n_layers = len(layers)

    saved_results = None
    with model.trace(prompt, remote=remote):
        # First collect all probs and top-k for each layer
        all_probs = []  # [n_layers, seq_len, vocab]
        all_top_indices = []  # [n_layers, seq_len, k]
        all_top_probs = []  # [n_layers, seq_len, k]

        for layer_idx in layers:
            hidden = model.model.layers[layer_idx].output[0]
            logits = model.lm_head(model.model.norm(hidden))
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
            for li in range(n_layers):
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
