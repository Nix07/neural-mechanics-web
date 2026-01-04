# Logit Lens Data Collection: Design and Rationale

This document explains the design of `logit_lens_data.py`, which provides efficient data collection for logit lens visualization from transformer language models via NDIF remote execution.

## The Bandwidth Problem

When analyzing large language models remotely, bandwidth between the NDIF server and client is the primary bottleneck. Consider the naive approach for a typical analysis:

**Llama 3.1 70B parameters:**
- 80 transformer layers
- 128,000 vocabulary size
- 20 input tokens (typical prompt)

**Naive approach: Download full logits**
```
80 layers × 20 positions × 128,000 vocab × 4 bytes = 819 MB per prompt
```

This is clearly impractical for interactive visualization. Even with fast connections, users would wait minutes per prompt.

## Solution: Server-Side Reduction

The key insight is that visualization only needs a tiny subset of the data:
- Top-k predictions at each layer (typically k=5)
- Probability trajectories only for tokens that appear in top-k somewhere

By performing reduction operations on the server before transmission, we achieve:

**Optimized approach:**
```
Top-k indices + probs: 80 × 20 × 5 × 8 bytes = 64 KB
Trajectories (~50 tracked tokens): 80 × 20 × 50 × 4 bytes = 320 KB
Total: ~400 KB (2000× reduction)
```

## Two Collection Modes

### Mode 1: `track_across_layers=False`

Simple top-k extraction at each layer, no trajectories.

```python
data = collect_logit_lens_topk_efficient(prompt, model, top_k=5)
# Returns:
#   top_indices: [80, 20, 5] - which tokens are top-5
#   top_probs: [80, 20, 5] - their probabilities
```

**Use case:** Quick analysis, heatmap visualization without trajectory lines.

**Bandwidth:** ~64 KB for the example above.

### Mode 2: `track_across_layers=True`

Identifies all tokens appearing in top-k at ANY layer, then extracts their full probability trajectory across all layers.

```python
data = collect_logit_lens_topk_efficient(
    prompt, model, top_k=5, track_across_layers=True
)
# Returns additionally:
#   tracked_indices: List of unique token sets per position
#   tracked_probs: [n_layers, n_tracked] probability matrices
```

**Use case:** Interactive trajectory visualization showing how predictions evolve through layers.

**Bandwidth:** ~400 KB for the example above.

## Multi-Model Support

Different transformer architectures organize their layers, normalization, and output head differently. The module uses a registry pattern to support multiple model families.

### Supported Models

| Model Type | Layers Path | Norm | lm_head | Examples |
|------------|-------------|------|---------|----------|
| `llama` | `model.layers` | `model.norm` | `lm_head` | Llama 2, Llama 3, CodeLlama |
| `mistral` | `model.layers` | `model.norm` | `lm_head` | Mistral, Mixtral |
| `qwen2` | `model.layers` | `model.norm` | `lm_head` | Qwen 2 |
| `gpt2` | `transformer.h` | `transformer.ln_f` | `lm_head` | GPT-2 |
| `gptj` | `transformer.h` | `transformer.ln_f` | `lm_head` | GPT-J |
| `gpt_neox` | `gpt_neox.layers` | `gpt_neox.final_layer_norm` | `embed_out` | GPT-NeoX, Pythia |
| `olmo` | `model.transformer.blocks` | `model.transformer.ln_f` | `model.transformer.ff_out` | OLMo |
| `phi` | `model.layers` | `model.final_layernorm` | `lm_head` | Phi-3 |
| `gemma` | `model.layers` | `model.norm` | `lm_head` | Gemma, Gemma 2 |

### Auto-Detection

The model type is automatically detected from `model.config.model_type`:

```python
# Auto-detection (recommended)
data = collect_logit_lens_topk_efficient(prompt, model)

# Explicit model type (if auto-detection fails)
data = collect_logit_lens_topk_efficient(prompt, model, model_type="llama")
```

### Registry Pattern

The `MODEL_CONFIGS` dictionary maps model types to accessor specifications:

```python
MODEL_CONFIGS = {
    "llama": {
        "layers": "model.layers",       # Path to layer list
        "norm": "model.norm",           # Final layer norm
        "lm_head": "lm_head",           # Output projection
        "n_layers": "config.num_hidden_layers",  # Layer count
    },
    # ... more models
}
```

### Custom Models

For models not in the registry, you can add entries with either string paths or callables:

```python
from logit_lens_data import MODEL_CONFIGS

# String path approach (simple)
MODEL_CONFIGS["my_model"] = {
    "layers": "backbone.blocks",
    "norm": "backbone.final_norm",
    "lm_head": "output_projection",
    "n_layers": "config.num_layers",
}

# Callable approach (flexible)
MODEL_CONFIGS["custom_model"] = {
    "layers": lambda m: m.get_layers(),
    "norm": lambda m, hidden: m.apply_norm(hidden),  # 2-arg callable
    "lm_head": lambda m: m.get_lm_head(),
    "n_layers": lambda m: len(m.get_layers()),
}
```

Callables support two signatures:
- **1-arg** `fn(model)` - returns a module or weight matrix to apply
- **2-arg** `fn(model, hidden)` - applies transformation directly to hidden state

### Helper Functions

```python
from logit_lens_data import (
    get_model_config,    # Get config dict, auto-detect if needed
    detect_model_type,   # Just detect type from config
    resolve_accessor,    # Resolve string/callable to value
)

# Get config
cfg = get_model_config(model)  # Auto-detect
cfg = get_model_config(model, model_type="gpt2")  # Explicit

# Access model components
layers = resolve_accessor(model, cfg["layers"])
n_layers = resolve_accessor(model, cfg["n_layers"])
```

## Implementation Strategy

### nnsight Trace Context

All computation happens within a single `model.trace()` context:

```python
cfg = get_model_config(model)  # Auto-detect architecture
model_layers = resolve_accessor(model, cfg["layers"])

with model.trace(prompt, remote=True):
    # Build computation graph
    for layer_idx in layers:
        hidden = model_layers[layer_idx].output[0]
        normed = apply_module_or_callable(model, cfg["norm"], hidden)
        logits = apply_module_or_callable(model, cfg["lm_head"], normed)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        # ... more processing

    # Save only what we need
    saved_results = (top_indices, top_probs, tracked_data).save()
```

The `.save()` call marks tensors for transmission back to the client. Everything else stays on the server.

### Trajectory Extraction Algorithm

For each input position:

1. **Collect candidates:** Gather all token indices appearing in top-k at any layer
   ```python
   pos_indices = stacked_top_indices[:, pos, :].reshape(-1)  # [n_layers * k]
   unique_tokens = torch.unique(pos_indices)  # typically 20-100 tokens
   ```

2. **Extract probabilities:** Index into each layer's probability distribution
   ```python
   for layer in range(n_layers):
       probs_at_layer = all_probs[layer][pos, unique_tokens]
   ```

3. **Save compact result:** Only unique indices and their trajectories
   ```python
   tracked_results.append((unique_tokens, stacked_probs))
   ```

This approach is efficient because:
- `unique_tokens` is typically 20-100 tokens (not 128,000)
- Indexing is fast on GPU
- Only small tensors cross the network

## Data Format

The returned dictionary matches what `LogitLensWidget` expects:

```python
{
    "tokens": ["The", " quick", " brown", ...],  # Input tokens
    "layers": [0, 1, 2, ..., 79],                # Layer indices
    "top_indices": Tensor[80, 20, 5],            # Top-k token IDs
    "top_probs": Tensor[80, 20, 5],              # Top-k probabilities

    # If track_across_layers=True:
    "tracked_indices": [Tensor[n0], Tensor[n1], ...],  # Per-position
    "tracked_probs": [Tensor[80, n0], Tensor[80, n1], ...]  # Trajectories
}
```

### Converting to Widget JSON

The `fetch_preview_data.py` script shows how to convert this to the JSON format expected by `LogitLensWidget`:

```python
js_data = {
    "layers": data["layers"],
    "tokens": data["tokens"],
    "cells": []  # [position][layer] structure
}

for pos in range(len(data["tokens"])):
    pos_data = []
    for layer in range(len(data["layers"])):
        cell = {
            "token": tokenizer.decode([top_indices[layer, pos, 0]]),
            "prob": top_probs[layer, pos, 0].item(),
            "trajectory": [...],  # from tracked_probs
            "topk": [...]         # top-k with trajectories
        }
        pos_data.append(cell)
    js_data["cells"].append(pos_data)
```

## Comparison: Naive vs Efficient

### `collect_logit_lens_topk` (Naive)

```python
cfg = get_model_config(model)
model_layers = resolve_accessor(model, cfg["layers"])

with model.trace(prompt, remote=remote):
    logits_list = []
    for layer_idx in layers:
        hidden = model_layers[layer_idx].output[0]
        normed = apply_module_or_callable(model, cfg["norm"], hidden)
        logits = apply_module_or_callable(model, cfg["lm_head"], normed)
        logits_list.append(logits)
    saved_logits = logits_list.save()  # Downloads FULL logits!

# Client-side processing
all_logits = torch.stack(saved_logits)  # 819 MB received
all_probs = torch.softmax(all_logits, dim=-1)
top_probs, top_indices = all_probs.topk(top_k, dim=-1)
```

**Problems:**
- Downloads 819 MB of logits
- Client must have GPU memory for softmax
- Slow, wasteful

### `collect_logit_lens_topk_efficient` (Optimized)

```python
with model.trace(prompt, remote=remote):
    for layer_idx in layers:
        hidden = model_layers[layer_idx].output[0]
        normed = apply_module_or_callable(model, cfg["norm"], hidden)
        logits = apply_module_or_callable(model, cfg["lm_head"], normed)
        probs = torch.softmax(logits, dim=-1)      # On server
        top_p, top_i = probs.topk(top_k, dim=-1)   # On server
        # ... trajectory extraction on server
    saved_results = (top_indices, top_probs, tracked).save()

# Only ~400 KB received
top_indices, top_probs, tracked = saved_results
```

**Benefits:**
- Downloads only 400 KB
- No GPU needed on client
- Fast, interactive

## Layer Subset Analysis

Both functions accept a `layers` parameter to analyze only specific layers:

```python
# Every 4th layer for quick overview
data = collect_logit_lens_topk_efficient(
    prompt, model,
    layers=list(range(0, 80, 4))  # [0, 4, 8, ..., 76]
)
```

This further reduces bandwidth and is useful for:
- Quick exploration before detailed analysis
- Matching widget stride display (e.g., "showing every 4 layers")
- Comparing specific layers of interest

## Error Handling

The `get_value()` helper handles nnsight's proxy objects:

```python
def get_value(saved):
    try:
        return saved.value
    except AttributeError:
        return saved
```

This allows the same code to work with both remote (proxy) and local (direct tensor) execution.

## Performance Tips

1. **Use remote=True** for large models (>7B parameters)
2. **Start with track_across_layers=False** for initial exploration
3. **Reduce top_k** if bandwidth is constrained (k=3 often sufficient)
4. **Use layer subsets** for very long sequences
5. **Cache results** - the data is JSON-serializable for storage

## Files

- `logit_lens_data.py` - Core collection functions
- `fetch_preview_data.py` - Example script for fetching and converting data
- `preview_data.js` - Cached example data (JSONP format for browser)
