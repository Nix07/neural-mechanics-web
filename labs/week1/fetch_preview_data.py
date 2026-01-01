"""Fetch real data from NDIF for preview HTML."""
import json
import sys
import os
sys.path.insert(0, '.')

# Load tokens from .env.local
env_path = os.path.join(os.path.dirname(__file__), '../../.env.local')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                # Strip quotes if present
                val = val.strip('"').strip("'")
                os.environ[key] = val

import nnsight
from nnsight import LanguageModel, CONFIG
from logit_lens_data import collect_logit_lens_topk_efficient
import torch

# Set NDIF API key
if 'NDIF_API' in os.environ:
    CONFIG.set_default_api_key(os.environ['NDIF_API'])
    print("NDIF API key configured")

# Set HuggingFace token
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    print("HF token configured")

print("Loading model...")
# Use Llama 3.1 70B via NDIF
model = LanguageModel("meta-llama/Llama-3.1-70B", device_map="auto", token=hf_token)

# A pun that plays on "current" (electrical vs water)
prompt = "Why do electricians make good swimmers? Because they know the"
print(f"Collecting data for: {prompt}")

data = collect_logit_lens_topk_efficient(
    prompt, model,
    top_k=5,
    track_across_layers=True,  # Falls back to full version for this, but we need trajectories
    remote=True
)

# Convert to JSON-serializable format
js_data = {
    "layers": data["layers"],
    "tokens": data["tokens"],
    "cells": []
}

tokenizer = model.tokenizer
n_layers_total = len(data["layers"])

for pos in range(len(data["tokens"])):
    pos_data = []
    tracked_idx_list = data["tracked_indices"][pos].tolist()
    tracked_probs_matrix = data["tracked_probs"][pos]

    for li in range(n_layers_total):
        top_p = data["top_probs"][li, pos]
        top_i = data["top_indices"][li, pos]

        top1_idx = top_i[0].item()
        top1_tok = tokenizer.decode([top1_idx])
        top1_prob = top_p[0].item()

        if top1_idx in tracked_idx_list:
            ti = tracked_idx_list.index(top1_idx)
            top1_trajectory = tracked_probs_matrix[:, ti].tolist()
        else:
            top1_trajectory = [0.0] * n_layers_total

        topk_list = []
        for ki in range(5):
            tok_idx = top_i[ki].item()
            tok_str = tokenizer.decode([tok_idx])
            tok_prob = top_p[ki].item()

            if tok_idx in tracked_idx_list:
                ti = tracked_idx_list.index(tok_idx)
                trajectory = tracked_probs_matrix[:, ti].tolist()
            else:
                trajectory = [0.0] * n_layers_total

            topk_list.append({
                "token": tok_str,
                "prob": round(tok_prob, 5),
                "trajectory": [round(p, 5) for p in trajectory]
            })

        pos_data.append({
            "token": top1_tok,
            "prob": round(top1_prob, 5),
            "trajectory": [round(p, 5) for p in top1_trajectory],
            "topk": topk_list
        })

    js_data["cells"].append(pos_data)

# Save as JSONP for loading via file:// protocol
with open("preview_data.js", "w") as f:
    f.write("var PREVIEW_DATA = ")
    json.dump(js_data, f, separators=(',', ':'))
    f.write(";")

print(f"Saved data with {len(js_data['tokens'])} tokens, {len(js_data['layers'])} layers")
print("Data saved to preview_data.js")
