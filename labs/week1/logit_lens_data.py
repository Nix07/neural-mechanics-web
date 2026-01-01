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

    # Track across layers mode - need two passes or smarter approach
    # First pass: get top-k indices at each layer
    # Second pass: would need to extract specific probs (complex for remote)
    # For now, fall back to full logits approach for this mode
    return collect_logit_lens_topk(
        prompt, model, top_k, track_across_layers=True, remote=remote, layers=layers
    )


def decode_tracked_tokens(data: Dict, tokenizer) -> Dict[int, List[str]]:
    """
    Decode tracked token indices to strings.

    Returns: {position: [token_str, ...]}
    """
    result = {}
    for pos, indices in enumerate(data["tracked_indices"]):
        result[pos] = [tokenizer.decode([idx.item()]) for idx in indices]
    return result
