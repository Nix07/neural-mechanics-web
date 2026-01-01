"""
Interactive Logit Lens Visualizations using HTML/CSS

Creates beautiful, interactive heatmaps with hover tooltips showing
top-k predictions at each layer and position.

Usage:
    from logit_lens_viz import logit_lens_heatmap, token_trajectory
    from logit_lens_viz import render_trajectory_heatmap

    # Heatmap with hover showing top predictions
    logit_lens_heatmap(prompt, model, target_token=" Paris", remote=True)

    # Efficient trajectory visualization (uses logit_lens_data)
    from logit_lens_data import collect_logit_lens_topk
    data = collect_logit_lens_topk(prompt, model, top_k=5, track_across_layers=True)
    render_trajectory_heatmap(data, model.tokenizer, position=-1)
"""

import torch
import numpy as np
from IPython.display import HTML, display
import html


def get_value(saved):
    """Helper to get value from saved tensor."""
    try:
        return saved.value
    except AttributeError:
        return saved


def collect_logit_lens_data(prompt, model, remote=True):
    """Collect logit lens data for all layers and positions.

    Uses list.save() to save the list of logits directly, which works
    with both local and NDIF remote execution.
    """
    n_layers = model.config.num_hidden_layers
    tokens = model.tokenizer.encode(prompt)
    token_strs = [model.tokenizer.decode([t]) for t in tokens]

    saved_logits = None
    with model.trace(prompt, remote=remote):
        logits_list = []
        for layer_idx in range(n_layers):
            hidden = model.model.layers[layer_idx].output[0]
            logits = model.lm_head(model.model.norm(hidden))
            seq_logits = logits[0] if len(logits.shape) == 3 else logits
            logits_list.append(seq_logits)
        # Save the list directly - returns list of tensors after trace
        saved_logits = logits_list.save()

    # Stack after trace for consistent shape [n_layers, seq_len, vocab]
    stacked_logits = torch.stack([get_value(t).float() for t in saved_logits])
    return stacked_logits, token_strs


def prob_to_color(prob, colormap='blues'):
    """Convert probability [0,1] to RGB color string."""
    if colormap == 'blues':
        # White to blue gradient
        r = int(255 * (1 - prob * 0.8))
        g = int(255 * (1 - prob * 0.6))
        b = 255
    elif colormap == 'viridis':
        # Approximate viridis
        r = int(68 + (253 - 68) * (1 - prob))
        g = int(1 + (231 - 1) * prob * 0.8)
        b = int(84 + (37 - 84) * prob)
    else:
        # Grayscale
        v = int(255 * (1 - prob))
        r, g, b = v, v, v
    return f'rgb({r},{g},{b})'


def logit_lens_heatmap(prompt, model, target_token=None, remote=True,
                       top_k=5, layer_step=1, cell_size=24):
    """
    Display an interactive logit lens heatmap.

    Layout: Input tokens on left margin (top to bottom), layers as columns (left to right).
    The rightmost column shows predictions from the final layer.
    Hover over cells to see top-k predictions.

    Args:
        prompt: Input text
        model: nnsight LanguageModel
        target_token: Token to track (colors show its probability)
        remote: Use NDIF remote execution
        top_k: Number of predictions to show on hover
        layer_step: Show every Nth layer (for large models)
        cell_size: Size of each cell in pixels
    """
    # Collect data
    stacked_logits, token_strs = collect_logit_lens_data(prompt, model, remote)
    n_layers, seq_len, vocab_size = stacked_logits.shape

    # Subsample layers if needed
    layer_indices = list(range(0, n_layers, layer_step))
    if (n_layers - 1) not in layer_indices:
        layer_indices.append(n_layers - 1)

    # Get target token ID if specified
    if target_token:
        target_ids = model.tokenizer.encode(target_token, add_special_tokens=False)
        target_id = target_ids[0]
    else:
        target_id = None

    # Build HTML - rows are token positions, columns are layers
    rows_html = []

    for pos in range(seq_len):
        cells = []
        # First cell: input token label
        tok_display = html.escape(token_strs[pos])
        cells.append(f'<td class="token-label" title="{tok_display}">{tok_display}</td>')

        # One cell per layer
        for layer_idx in layer_indices:
            logits = stacked_logits[layer_idx, pos]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k)

            # Cell color based on target token or top prediction
            if target_id is not None:
                cell_prob = probs[target_id].item()
            else:
                cell_prob = top_probs[0].item()

            color = prob_to_color(cell_prob)

            # Build tooltip content
            tooltip_lines = [f'Layer {layer_idx}']
            for p, idx in zip(top_probs, top_indices):
                tok = model.tokenizer.decode([idx])
                tok_escaped = html.escape(repr(tok)[1:-1])
                prob_pct = p.item() * 100
                tooltip_lines.append(f'{tok_escaped}: {prob_pct:.1f}%')

            tooltip = '&#10;'.join(tooltip_lines)

            cells.append(
                f'<td class="heatmap-cell" style="background:{color};" '
                f'title="{tooltip}"></td>'
            )

        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    # Layer header row (at bottom, as footer)
    layer_headers = ''.join(
        f'<th class="layer-label">{layer_idx}</th>'
        for layer_idx in layer_indices
    )
    footer_row = f'<tr><th class="corner-label">Layer</th>{layer_headers}</tr>'

    # Title
    if target_token:
        title = f'Logit Lens: P("{html.escape(target_token)}")'
    else:
        title = 'Logit Lens: Top Prediction Probability'

    subtitle = html.escape(prompt[:80]) + ('...' if len(prompt) > 80 else '')

    full_html = f'''
    <style>
        .logit-lens-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px 0;
        }}
        .logit-lens-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .logit-lens-subtitle {{
            font-size: 12px;
            color: #666;
            margin-bottom: 12px;
            font-style: italic;
        }}
        .logit-lens-table {{
            border-collapse: collapse;
            font-size: 11px;
        }}
        .logit-lens-table td, .logit-lens-table th {{
            padding: 0;
            border: 1px solid #e0e0e0;
        }}
        .heatmap-cell {{
            width: {cell_size}px;
            height: {cell_size}px;
        }}
        .heatmap-cell:hover {{
            outline: 2px solid #333;
            outline-offset: -1px;
            cursor: pointer;
        }}
        .token-label {{
            padding: 2px 8px !important;
            text-align: right;
            font-weight: 500;
            color: #333;
            background: #f8f8f8;
            border: none !important;
            white-space: nowrap;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .layer-label {{
            padding: 4px 2px !important;
            text-align: center;
            font-weight: 500;
            color: #666;
            background: #f8f8f8;
            width: {cell_size}px;
        }}
        .corner-label {{
            padding: 4px 8px !important;
            text-align: right;
            font-weight: 500;
            color: #666;
            background: #f8f8f8;
            border: none !important;
        }}
    </style>
    <div class="logit-lens-container">
        <div class="logit-lens-title">{title}</div>
        <div class="logit-lens-subtitle">{subtitle}</div>
        <table class="logit-lens-table">
            {"".join(rows_html)}
            {footer_row}
        </table>
    </div>
    '''

    display(HTML(full_html))


def token_trajectory(prompt, target_tokens, model, remote=True, height=200):
    """
    Display how token probabilities evolve across layers using SVG.

    Args:
        prompt: Input text
        target_tokens: Token or list of tokens to track
        model: nnsight LanguageModel
        remote: Use NDIF remote execution
        height: Chart height in pixels
    """
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens]

    stacked_logits, token_strs = collect_logit_lens_data(prompt, model, remote)
    n_layers = stacked_logits.shape[0]

    # Collect probabilities for each target token
    all_probs = {}
    for target_token in target_tokens:
        target_ids = model.tokenizer.encode(target_token, add_special_tokens=False)
        target_id = target_ids[0]

        layer_probs = []
        for layer_idx in range(n_layers):
            probs = torch.softmax(stacked_logits[layer_idx, -1], dim=-1)
            layer_probs.append(probs[target_id].item())
        all_probs[target_token] = layer_probs

    # Find max probability for scaling
    max_prob = max(max(probs) for probs in all_probs.values())
    max_prob = max(max_prob, 0.1)  # At least 0.1 for scale

    # SVG dimensions
    width = 600
    margin = {'top': 20, 'right': 100, 'bottom': 40, 'left': 50}
    inner_width = width - margin['left'] - margin['right']
    inner_height = height - margin['top'] - margin['bottom']

    # Colors for different tokens
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    # Build SVG paths
    paths_html = []
    legend_html = []

    for i, (target_token, layer_probs) in enumerate(all_probs.items()):
        color = colors[i % len(colors)]

        # Create path
        points = []
        for layer_idx, prob in enumerate(layer_probs):
            x = margin['left'] + (layer_idx / (n_layers - 1)) * inner_width
            y = margin['top'] + inner_height - (prob / max_prob) * inner_height
            points.append(f'{x:.1f},{y:.1f}')

        path_d = 'M ' + ' L '.join(points)
        paths_html.append(
            f'<path d="{path_d}" stroke="{color}" stroke-width="2" fill="none"/>'
        )

        # Add dots
        for layer_idx, prob in enumerate(layer_probs):
            x = margin['left'] + (layer_idx / (n_layers - 1)) * inner_width
            y = margin['top'] + inner_height - (prob / max_prob) * inner_height
            paths_html.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}">'
                f'<title>Layer {layer_idx}: {prob:.4f}</title></circle>'
            )

        # Legend entry
        legend_y = margin['top'] + 15 + i * 18
        tok_escaped = html.escape(target_token)
        legend_html.append(
            f'<line x1="{width - margin["right"] + 10}" y1="{legend_y}" '
            f'x2="{width - margin["right"] + 25}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<text x="{width - margin["right"] + 30}" y="{legend_y + 4}" '
            f'font-size="11" fill="#333">{tok_escaped}</text>'
        )

    # X axis
    x_axis = f'''
        <line x1="{margin['left']}" y1="{height - margin['bottom']}"
              x2="{width - margin['right']}" y2="{height - margin['bottom']}"
              stroke="#333" stroke-width="1"/>
        <text x="{width/2}" y="{height - 5}" text-anchor="middle"
              font-size="12" fill="#333">Layer</text>
    '''

    # Y axis
    y_axis = f'''
        <line x1="{margin['left']}" y1="{margin['top']}"
              x2="{margin['left']}" y2="{height - margin['bottom']}"
              stroke="#333" stroke-width="1"/>
        <text x="15" y="{height/2}" text-anchor="middle"
              font-size="12" fill="#333" transform="rotate(-90, 15, {height/2})">Probability</text>
    '''

    # Y axis ticks
    y_ticks = []
    for i in range(5):
        prob_val = max_prob * i / 4
        y = margin['top'] + inner_height - (i / 4) * inner_height
        y_ticks.append(
            f'<text x="{margin["left"] - 5}" y="{y + 3}" text-anchor="end" '
            f'font-size="10" fill="#666">{prob_val:.2f}</text>'
            f'<line x1="{margin["left"] - 3}" y1="{y}" x2="{margin["left"]}" y2="{y}" '
            f'stroke="#666"/>'
        )

    title = html.escape(prompt[:60]) + ('...' if len(prompt) > 60 else '')

    svg_html = f'''
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px 0;">
        <div style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">
            Token Probability Across Layers
        </div>
        <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">
            {title}
        </div>
        <svg width="{width}" height="{height}" style="background: #fafafa; border-radius: 4px;">
            {x_axis}
            {y_axis}
            {"".join(y_ticks)}
            {"".join(paths_html)}
            {"".join(legend_html)}
        </svg>
    </div>
    '''

    display(HTML(svg_html))


def render_trajectory_heatmap(data, tokenizer, position=-1, cell_width=28, cell_height=24,
                               show_top_k_border=True, max_tokens=20):
    """
    Render an interactive heatmap showing how tracked tokens' probabilities
    evolve across layers.

    Args:
        data: Output from collect_logit_lens_topk with track_across_layers=True
        tokenizer: Model tokenizer for decoding token IDs
        position: Token position to visualize (-1 for last)
        cell_width: Width of each layer cell
        cell_height: Height of each token row
        show_top_k_border: Highlight cells where token is in top-k
        max_tokens: Maximum number of tracked tokens to display

    Layout:
        - Rows: tracked output tokens (sorted by max probability)
        - Columns: layers (left to right)
        - Color: probability intensity
        - Border: indicates top-k at that layer
    """
    if "tracked_probs" not in data:
        raise ValueError("Data must be from collect_logit_lens_topk with track_across_layers=True")

    seq_len = len(data["tokens"])
    if position < 0:
        position = seq_len + position

    layers = data["layers"]
    tracked_indices = data["tracked_indices"][position]
    tracked_probs = data["tracked_probs"][position]  # [n_layers, n_tracked]
    top_indices = data["top_indices"][:, position, :]  # [n_layers, k]

    n_layers = len(layers)
    n_tracked = len(tracked_indices)

    # Decode token strings
    token_strs = [tokenizer.decode([idx.item()]) for idx in tracked_indices]

    # Sort by maximum probability across layers (descending)
    max_probs = tracked_probs.max(dim=0).values
    sorted_order = torch.argsort(max_probs, descending=True)

    # Limit number of tokens shown
    if n_tracked > max_tokens:
        sorted_order = sorted_order[:max_tokens]

    # Build rows
    rows_html = []
    for rank, tok_idx in enumerate(sorted_order):
        tok_str = token_strs[tok_idx]
        tok_display = html.escape(repr(tok_str)[1:-1])  # Show escaped repr

        cells = []
        # Token label cell
        cells.append(
            f'<td class="tok-label" title="{html.escape(tok_str)}">{tok_display}</td>'
        )

        # One cell per layer
        for layer_i, layer_idx in enumerate(layers):
            prob = tracked_probs[layer_i, tok_idx].item()
            color = prob_to_color(prob, colormap='blues')

            # Check if this token is in top-k at this layer
            is_topk = tracked_indices[tok_idx] in top_indices[layer_i]

            border_style = "border: 2px solid #e91e63;" if (is_topk and show_top_k_border) else ""

            tooltip = f"Layer {layer_idx}: {prob*100:.2f}%"
            cells.append(
                f'<td class="heat-cell" style="background:{color};{border_style}" '
                f'title="{tooltip}"></td>'
            )

        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    # Layer header row (footer)
    layer_headers = ''.join(
        f'<th class="layer-hdr">{l}</th>' for l in layers
    )
    footer_row = f'<tr><th class="corner">Layer</th>{layer_headers}</tr>'

    # Title
    input_token = data["tokens"][position]
    title = f'Token Trajectories at Position {position}: "{html.escape(input_token)}"'
    subtitle = f'Showing {len(sorted_order)} tokens that appear in top-k at any layer'

    table_width = 120 + len(layers) * cell_width

    full_html = f'''
    <style>
        .trajectory-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px 0;
        }}
        .trajectory-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .trajectory-subtitle {{
            font-size: 12px;
            color: #666;
            margin-bottom: 12px;
        }}
        .trajectory-table {{
            border-collapse: collapse;
            font-size: 11px;
        }}
        .trajectory-table td, .trajectory-table th {{
            padding: 0;
        }}
        .heat-cell {{
            width: {cell_width}px;
            height: {cell_height}px;
            border: 1px solid #e0e0e0;
        }}
        .heat-cell:hover {{
            outline: 2px solid #333;
            outline-offset: -1px;
            cursor: pointer;
        }}
        .tok-label {{
            padding: 2px 8px;
            text-align: right;
            font-weight: 500;
            color: #333;
            background: #f8f8f8;
            white-space: nowrap;
            max-width: 100px;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: monospace;
            font-size: 10px;
        }}
        .layer-hdr {{
            padding: 4px 2px;
            text-align: center;
            font-weight: 500;
            color: #666;
            background: #f8f8f8;
            width: {cell_width}px;
            font-size: 9px;
        }}
        .corner {{
            padding: 4px 8px;
            text-align: right;
            font-weight: 500;
            color: #666;
            background: #f8f8f8;
        }}
        .legend {{
            margin-top: 8px;
            font-size: 11px;
            color: #666;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 16px;
        }}
        .legend-box {{
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 4px;
            vertical-align: middle;
            border: 2px solid #e91e63;
        }}
    </style>
    <div class="trajectory-container">
        <div class="trajectory-title">{title}</div>
        <div class="trajectory-subtitle">{subtitle}</div>
        <table class="trajectory-table">
            {"".join(rows_html)}
            {footer_row}
        </table>
        <div class="legend">
            <span class="legend-item">
                <span class="legend-box" style="background:#fff;"></span>
                Pink border = top-k at that layer
            </span>
        </div>
    </div>
    '''

    display(HTML(full_html))


def render_all_positions_heatmap(data, tokenizer, cell_width=48, cell_height=22, layer_step=1,
                                  chart_height=120, show_trajectory=True):
    """
    Render an interactive heatmap showing top predictions at all positions and layers.

    Each cell shows the top-1 predicted token (cropped to fit) with background
    color indicating probability.

    Interactions:
    - Hover: shows probability trajectory of that token across all layers in SVG below
    - Click: opens popup with full token and top-k list; hover top-k to see their trajectories

    Args:
        data: Output from collect_logit_lens_topk with track_across_layers=True
        tokenizer: Model tokenizer
        cell_width: Width of each layer column (text cropped to fit)
        cell_height: Height of each row
        layer_step: Show every Nth layer
        chart_height: Height of trajectory chart
        show_trajectory: Whether to show the trajectory chart below
    """
    import json
    import random

    # Generate unique ID for this instance
    uid = f"ll_{random.randint(10000, 99999)}"

    tokens = data["tokens"]
    layers = data["layers"]
    top_indices = data["top_indices"]  # [n_layers, seq_len, k]
    top_probs = data["top_probs"]  # [n_layers, seq_len, k]

    # Check if we have tracked data for trajectories
    has_tracked = "tracked_probs" in data and "tracked_indices" in data

    # Subsample layers for display
    layer_indices = list(range(0, len(layers), layer_step))
    if len(layers) - 1 not in layer_indices:
        layer_indices.append(len(layers) - 1)

    n_layers_display = len(layer_indices)
    n_layers_total = len(layers)
    seq_len = len(tokens)
    k = top_indices.shape[2]

    # Build data structure for JavaScript
    # For each position, store top-k token info and their probabilities across layers
    js_data = {
        "layers": layers,
        "layerIndices": layer_indices,
        "tokens": tokens,
        "cells": []  # [pos][layer_i] = {token, probs_across_layers, topk: [{token, prob, probs_across_layers}]}
    }

    for pos in range(seq_len):
        pos_data = []
        for li, layer_idx in enumerate(layer_indices):
            top_p = top_probs[li, pos]
            top_i = top_indices[li, pos]

            # Get top-1 token
            top1_idx = top_i[0].item()
            top1_tok = tokenizer.decode([top1_idx])
            top1_prob = top_p[0].item()

            # Get trajectory for top-1 token (prob at each layer)
            if has_tracked:
                # Find this token in tracked indices for this position
                tracked_idx_list = data["tracked_indices"][pos].tolist()
                tracked_probs_matrix = data["tracked_probs"][pos]  # [n_layers, n_tracked]
                if top1_idx in tracked_idx_list:
                    ti = tracked_idx_list.index(top1_idx)
                    top1_trajectory = tracked_probs_matrix[:, ti].tolist()
                else:
                    top1_trajectory = [0.0] * n_layers_total
            else:
                top1_trajectory = [0.0] * n_layers_total

            # Build top-k list
            topk_list = []
            for ki in range(k):
                tok_idx = top_i[ki].item()
                tok_str = tokenizer.decode([tok_idx])
                tok_prob = top_p[ki].item()

                # Get trajectory for this token
                if has_tracked:
                    if tok_idx in tracked_idx_list:
                        ti = tracked_idx_list.index(tok_idx)
                        trajectory = tracked_probs_matrix[:, ti].tolist()
                    else:
                        trajectory = [0.0] * n_layers_total
                else:
                    trajectory = [0.0] * n_layers_total

                topk_list.append({
                    "token": tok_str,
                    "prob": tok_prob,
                    "trajectory": trajectory
                })

            pos_data.append({
                "token": top1_tok,
                "prob": top1_prob,
                "trajectory": top1_trajectory,
                "topk": topk_list
            })

        js_data["cells"].append(pos_data)

    # Build HTML table
    rows_html = []
    for pos, tok in enumerate(tokens):
        cells = []
        tok_display = html.escape(tok)
        cells.append(f'<td class="input-token" title="{tok_display}">{tok_display}</td>')

        for li, layer_idx in enumerate(layer_indices):
            cell_data = js_data["cells"][pos][li]
            cell_prob = cell_data["prob"]
            color = prob_to_color(cell_prob)
            top1_display = html.escape(cell_data["token"])
            text_color = "#333" if cell_prob < 0.5 else "#fff"

            cells.append(
                f'<td class="pred-cell" data-pos="{pos}" data-li="{li}" '
                f'style="background:{color}; color:{text_color};">{top1_display}</td>'
            )

        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    # Footer with layer numbers
    layer_headers = ''.join(
        f'<th class="layer-hdr">{layers[li]}</th>' for li in layer_indices
    )
    footer_row = f'<tr><th class="corner-hdr">Layer</th>{layer_headers}</tr>'

    # Chart dimensions
    chart_width = 100 + n_layers_display * cell_width
    margin = {"top": 10, "right": 20, "bottom": 25, "left": 45}
    inner_width = chart_width - margin["left"] - margin["right"]
    inner_height = chart_height - margin["top"] - margin["bottom"]

    full_html = f'''
    <style>
        #{uid} {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px 0;
            position: relative;
        }}
        #{uid} .ll-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        #{uid} .ll-table {{
            border-collapse: collapse;
            font-size: 10px;
            table-layout: fixed;
        }}
        #{uid} .ll-table td, #{uid} .ll-table th {{
            border: 1px solid #ddd;
        }}
        #{uid} .pred-cell {{
            width: {cell_width}px;
            max-width: {cell_width}px;
            height: {cell_height}px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            padding: 2px 4px;
            font-family: monospace;
            font-size: 9px;
            cursor: pointer;
        }}
        #{uid} .pred-cell:hover {{
            outline: 2px solid #e91e63;
            outline-offset: -1px;
        }}
        #{uid} .pred-cell.selected {{
            outline: 3px solid #2196F3;
            outline-offset: -1px;
        }}
        #{uid} .input-token {{
            padding: 2px 8px;
            text-align: right;
            font-weight: 500;
            color: #333;
            background: #f5f5f5;
            white-space: nowrap;
            max-width: 100px;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: monospace;
            font-size: 10px;
        }}
        #{uid} .layer-hdr {{
            width: {cell_width}px;
            max-width: {cell_width}px;
            padding: 4px 2px;
            text-align: center;
            font-weight: 500;
            color: #666;
            background: #f5f5f5;
            font-size: 9px;
        }}
        #{uid} .corner-hdr {{
            padding: 4px 8px;
            text-align: right;
            font-weight: 500;
            color: #666;
            background: #f5f5f5;
        }}
        #{uid} .chart-container {{
            margin-top: 12px;
            background: #fafafa;
            border-radius: 4px;
            padding: 8px;
        }}
        #{uid} .chart-label {{
            font-size: 11px;
            color: #666;
            margin-bottom: 4px;
        }}
        #{uid} .popup {{
            display: none;
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 12px;
            z-index: 100;
            min-width: 180px;
            max-width: 280px;
        }}
        #{uid} .popup.visible {{
            display: block;
        }}
        #{uid} .popup-header {{
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid #eee;
        }}
        #{uid} .popup-token {{
            font-family: monospace;
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        #{uid} .popup-close {{
            position: absolute;
            top: 8px;
            right: 10px;
            cursor: pointer;
            color: #999;
            font-size: 16px;
        }}
        #{uid} .popup-close:hover {{
            color: #333;
        }}
        #{uid} .topk-item {{
            padding: 4px 6px;
            margin: 2px 0;
            border-radius: 3px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            font-size: 11px;
        }}
        #{uid} .topk-item:hover {{
            background: #e3f2fd;
        }}
        #{uid} .topk-item.active {{
            background: #bbdefb;
        }}
        #{uid} .topk-token {{
            font-family: monospace;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        #{uid} .topk-prob {{
            color: #666;
            margin-left: 8px;
        }}
        #{uid} .pred-cell.pinned {{
            box-shadow: inset 0 0 0 2px #4CAF50;
        }}
        #{uid} .topk-item.pinned {{
            background: #e8f5e9;
            border-left: 3px solid #4CAF50;
        }}
        #{uid} .topk-item.pinned:hover {{
            background: #c8e6c9;
        }}
        #{uid} .hint {{
            font-size: 10px;
            color: #999;
            margin-top: 6px;
            font-style: italic;
        }}
        #{uid} .resize-handle {{
            position: absolute;
            width: 8px;
            height: 100%;
            background: transparent;
            cursor: col-resize;
            right: 0;
            top: 0;
            z-index: 10;
        }}
        #{uid} .resize-handle:hover,
        #{uid} .resize-handle.dragging {{
            background: rgba(33, 150, 243, 0.3);
        }}
        #{uid} .table-wrapper {{
            position: relative;
            display: inline-block;
        }}
        #{uid} .resize-hint {{
            position: absolute;
            bottom: -18px;
            right: 0;
            font-size: 9px;
            color: #999;
        }}
    </style>

    <div id="{uid}">
        <div class="ll-title">Logit Lens: Top Predictions by Layer</div>
        <div class="table-wrapper">
            <table class="ll-table" id="{uid}_table">
                <!-- Table will be built dynamically -->
            </table>
            <div class="resize-handle" id="{uid}_resize"></div>
            <div class="resize-hint" id="{uid}_resize_hint">drag to resize</div>
        </div>

        <div class="chart-container">
            <div class="chart-label">Token probability across layers — hover to preview, shift+click to pin for comparison</div>
            <svg id="{uid}_chart" width="{chart_width}" height="{chart_height}">
                <g transform="translate({margin['left']},{margin['top']})">
                    <!-- X axis -->
                    <line x1="0" y1="{inner_height}" x2="{inner_width}" y2="{inner_height}" stroke="#ccc"/>
                    <text x="{inner_width/2}" y="{inner_height + 20}" text-anchor="middle" font-size="10" fill="#666">Layer</text>
                    <!-- Y axis -->
                    <line x1="0" y1="0" x2="0" y2="{inner_height}" stroke="#ccc"/>
                    <text x="-30" y="{inner_height/2}" text-anchor="middle" font-size="10" fill="#666" transform="rotate(-90,-30,{inner_height/2})">Prob</text>
                    <!-- Path will be drawn here -->
                    <path id="{uid}_path" fill="none" stroke="#2196F3" stroke-width="2"/>
                    <!-- Dots group -->
                    <g id="{uid}_dots"></g>
                </g>
            </svg>
        </div>

        <div class="popup" id="{uid}_popup">
            <span class="popup-close" onclick="document.getElementById('{uid}_popup').classList.remove('visible')">&times;</span>
            <div class="popup-header">
                Layer <span id="{uid}_popup_layer"></span>, Position <span id="{uid}_popup_pos"></span>
            </div>
            <div id="{uid}_popup_content"></div>
        </div>
    </div>

    <script>
    (function() {{
        const data = {json.dumps(js_data)};
        const uid = "{uid}";
        const nLayers = data.layers.length;
        const chartInnerWidth = {inner_width};
        const chartInnerHeight = {inner_height};
        const inputTokenWidth = 100;  // Width of input token column
        const minCellWidth = 30;
        const maxCellWidth = 200;
        const containerMaxWidth = 900;  // Max table width before striding kicks in

        let currentCellWidth = {cell_width};
        let currentStride = 1;

        // Persistent trajectories for comparison
        const pinnedTrajectories = new Map();
        const colors = ["#2196F3", "#e91e63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#F44336", "#8BC34A"];
        let colorIndex = 0;

        function getNextColor() {{
            const c = colors[colorIndex % colors.length];
            colorIndex++;
            return c;
        }}

        function escapeHtml(text) {{
            const div = document.createElement("div");
            div.textContent = text;
            return div.innerHTML;
        }}

        function probToColor(prob) {{
            const r = Math.round(255 * (1 - prob * 0.8));
            const g = Math.round(255 * (1 - prob * 0.6));
            return `rgb(${{r}},${{g}},255)`;
        }}

        // Calculate which layers to show given cell width
        function computeVisibleLayers(cellWidth) {{
            const availableWidth = containerMaxWidth - inputTokenWidth;
            const maxCols = Math.floor(availableWidth / cellWidth);

            if (maxCols >= nLayers) {{
                // All layers fit
                return {{ stride: 1, indices: data.layers.map((_, i) => i) }};
            }}

            // Need to stride - always include last layer
            const stride = Math.ceil(nLayers / maxCols);
            const indices = [];

            // Start from 0, step by stride, but always end with last
            for (let i = 0; i < nLayers - 1; i += stride) {{
                indices.push(i);
            }}
            // Always add last layer
            if (indices[indices.length - 1] !== nLayers - 1) {{
                indices.push(nLayers - 1);
            }}

            return {{ stride, indices }};
        }}

        // Build the table HTML
        function buildTable(cellWidth, visibleLayerIndices) {{
            const table = document.getElementById(uid + "_table");
            let html = "";

            // Data rows
            data.tokens.forEach((tok, pos) => {{
                html += "<tr>";
                html += `<td class="input-token" title="${{escapeHtml(tok)}}">${{escapeHtml(tok)}}</td>`;

                visibleLayerIndices.forEach((li, colIdx) => {{
                    const cellData = data.cells[pos][li];
                    const color = probToColor(cellData.prob);
                    const textColor = cellData.prob < 0.5 ? "#333" : "#fff";
                    const isPinned = pinnedTrajectories.has(cellData.token);

                    html += `<td class="pred-cell${{isPinned ? ' pinned' : ''}}" ` +
                        `data-pos="${{pos}}" data-li="${{li}}" data-col="${{colIdx}}" ` +
                        `style="background:${{color}}; color:${{textColor}}; width:${{cellWidth}}px; max-width:${{cellWidth}}px;">` +
                        `${{escapeHtml(cellData.token)}}</td>`;
                }});

                html += "</tr>";
            }});

            // Footer row with layer numbers
            html += "<tr>";
            html += `<th class="corner-hdr">Layer</th>`;
            visibleLayerIndices.forEach((li, colIdx) => {{
                const isLast = colIdx === visibleLayerIndices.length - 1;
                html += `<th class="layer-hdr" style="width:${{cellWidth}}px; max-width:${{cellWidth}}px;">${{data.layers[li]}}</th>`;
            }});
            html += "</tr>";

            table.innerHTML = html;

            // Re-attach event listeners
            attachCellListeners();

            // Update hint
            const hint = document.getElementById(uid + "_resize_hint");
            const stride = visibleLayerIndices.length < nLayers ?
                Math.ceil(nLayers / visibleLayerIndices.length) : 1;
            hint.textContent = stride > 1 ?
                `showing every ${{stride}} layers (drag to adjust)` :
                `showing all ${{nLayers}} layers`;
        }}

        function attachCellListeners() {{
            document.querySelectorAll(`#${{uid}} .pred-cell`).forEach(cell => {{
                const pos = parseInt(cell.dataset.pos);
                const li = parseInt(cell.dataset.li);
                const cellData = data.cells[pos][li];

                cell.addEventListener("mouseenter", () => {{
                    drawAllTrajectories(cellData.trajectory, "#999", cellData.token);
                }});

                cell.addEventListener("mouseleave", () => {{
                    drawAllTrajectories(null, null, null);
                }});

                cell.addEventListener("click", (e) => {{
                    e.stopPropagation();

                    if (e.shiftKey) {{
                        const wasPinned = !togglePinnedTrajectory(cellData.token, cellData.trajectory);
                        cell.classList.toggle("pinned", !wasPinned);
                        drawAllTrajectories(null, null, null);
                        return;
                    }}

                    document.querySelectorAll(`#${{uid}} .pred-cell.selected`).forEach(c => c.classList.remove("selected"));
                    cell.classList.add("selected");

                    showPopup(cell, pos, li, cellData);
                }});
            }});
        }}

        function showPopup(cell, pos, li, cellData) {{
            const popup = document.getElementById(uid + "_popup");
            const rect = cell.getBoundingClientRect();
            const containerRect = document.getElementById(uid).getBoundingClientRect();

            popup.style.left = (rect.left - containerRect.left + rect.width + 5) + "px";
            popup.style.top = (rect.top - containerRect.top) + "px";

            document.getElementById(uid + "_popup_layer").textContent = data.layers[li];
            document.getElementById(uid + "_popup_pos").textContent = pos + " (" + data.tokens[pos] + ")";

            let contentHtml = "";
            cellData.topk.forEach((item, ki) => {{
                const probPct = (item.prob * 100).toFixed(1);
                const isPinned = pinnedTrajectories.has(item.token);
                contentHtml += `<div class="topk-item${{isPinned ? ' pinned' : ''}}" data-ki="${{ki}}">`;
                contentHtml += `<span class="topk-token">${{escapeHtml(item.token)}}</span>`;
                contentHtml += `<span class="topk-prob">${{probPct}}%</span>`;
                contentHtml += `</div>`;
            }});

            const contentEl = document.getElementById(uid + "_popup_content");
            contentEl.innerHTML = contentHtml;

            contentEl.querySelectorAll(".topk-item").forEach(item => {{
                const ki = parseInt(item.dataset.ki);
                const tokData = cellData.topk[ki];

                item.addEventListener("mouseenter", () => {{
                    contentEl.querySelectorAll(".topk-item").forEach(it => it.classList.remove("active"));
                    item.classList.add("active");
                    drawAllTrajectories(tokData.trajectory, "#999", tokData.token);
                }});

                item.addEventListener("mouseleave", () => {{
                    drawAllTrajectories(null, null, null);
                }});

                item.addEventListener("click", (e) => {{
                    if (e.shiftKey) {{
                        e.stopPropagation();
                        const wasPinned = !togglePinnedTrajectory(tokData.token, tokData.trajectory);
                        item.classList.toggle("pinned", !wasPinned);
                        drawAllTrajectories(null, null, null);
                    }}
                }});
            }});

            popup.classList.add("visible");
            drawAllTrajectories(cellData.trajectory, "#999", cellData.token);
        }}

        function togglePinnedTrajectory(token, trajectory) {{
            if (pinnedTrajectories.has(token)) {{
                pinnedTrajectories.delete(token);
                return false;
            }} else {{
                pinnedTrajectories.set(token, {{ trajectory, color: getNextColor() }});
                return true;
            }}
        }}

        function drawAllTrajectories(hoverTrajectory, hoverColor, hoverLabel) {{
            const svg = document.getElementById(uid + "_chart").querySelector("g");
            svg.querySelectorAll(".traj-path, .traj-dots, .traj-legend").forEach(el => el.remove());

            let allProbs = [];
            pinnedTrajectories.forEach((v) => allProbs.push(...v.trajectory));
            if (hoverTrajectory) allProbs.push(...hoverTrajectory);
            const maxProb = Math.max(...allProbs, 0.1);

            let legendY = 5;
            pinnedTrajectories.forEach((v, label) => {{
                drawSingleTrajectory(svg, v.trajectory, v.color, maxProb, label, false);
                const legend = document.createElementNS("http://www.w3.org/2000/svg", "g");
                legend.classList.add("traj-legend");
                legend.innerHTML = `
                    <line x1="${{chartInnerWidth + 5}}" y1="${{legendY}}" x2="${{chartInnerWidth + 20}}" y2="${{legendY}}" stroke="${{v.color}}" stroke-width="2"/>
                    <text x="${{chartInnerWidth + 25}}" y="${{legendY + 4}}" font-size="9" fill="#333">${{escapeHtml(label.slice(0, 8))}}</text>
                `;
                svg.appendChild(legend);
                legendY += 14;
            }});

            if (hoverTrajectory) {{
                drawSingleTrajectory(svg, hoverTrajectory, hoverColor || "#999", maxProb, hoverLabel, true);
            }}
        }}

        function drawSingleTrajectory(svg, trajectory, color, maxProb, label, isHover) {{
            if (!trajectory || trajectory.length === 0) return;

            const pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
            pathEl.classList.add("traj-path");
            if (isHover) pathEl.style.opacity = "0.7";

            let d = "";
            trajectory.forEach((p, i) => {{
                const x = (i / (nLayers - 1)) * chartInnerWidth;
                const y = chartInnerHeight - (p / maxProb) * chartInnerHeight;
                d += (i === 0 ? "M" : "L") + x + "," + y;
            }});

            pathEl.setAttribute("d", d);
            pathEl.setAttribute("fill", "none");
            pathEl.setAttribute("stroke", color);
            pathEl.setAttribute("stroke-width", isHover ? "1.5" : "2");
            if (isHover) pathEl.setAttribute("stroke-dasharray", "4,2");
            svg.appendChild(pathEl);

            const dotsG = document.createElementNS("http://www.w3.org/2000/svg", "g");
            dotsG.classList.add("traj-dots");
            trajectory.forEach((p, i) => {{
                const x = (i / (nLayers - 1)) * chartInnerWidth;
                const y = chartInnerHeight - (p / maxProb) * chartInnerHeight;
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", x);
                circle.setAttribute("cy", y);
                circle.setAttribute("r", isHover ? 2 : 3);
                circle.setAttribute("fill", color);
                if (isHover) circle.style.opacity = "0.7";
                const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
                title.textContent = (label || "") + " L" + i + ": " + (p * 100).toFixed(2) + "%";
                circle.appendChild(title);
                dotsG.appendChild(circle);
            }});
            svg.appendChild(dotsG);
        }}

        // Resize handle drag logic
        const resizeHandle = document.getElementById(uid + "_resize");
        let isDragging = false;
        let startX = 0;
        let startWidth = currentCellWidth;

        resizeHandle.addEventListener("mousedown", (e) => {{
            isDragging = true;
            startX = e.clientX;
            startWidth = currentCellWidth;
            resizeHandle.classList.add("dragging");
            e.preventDefault();
        }});

        document.addEventListener("mousemove", (e) => {{
            if (!isDragging) return;

            // Dragging left = negative delta = increase width (fewer columns)
            const delta = startX - e.clientX;
            const newWidth = Math.max(minCellWidth, Math.min(maxCellWidth, startWidth + delta * 0.5));

            if (Math.abs(newWidth - currentCellWidth) > 2) {{
                currentCellWidth = newWidth;
                const {{ indices }} = computeVisibleLayers(currentCellWidth);
                buildTable(currentCellWidth, indices);
            }}
        }});

        document.addEventListener("mouseup", () => {{
            if (isDragging) {{
                isDragging = false;
                resizeHandle.classList.remove("dragging");
            }}
        }});

        // Close popup when clicking outside
        document.addEventListener("click", (e) => {{
            if (!e.target.closest(`#${{uid}} .popup`) && !e.target.closest(`#${{uid}} .pred-cell`)) {{
                document.getElementById(uid + "_popup").classList.remove("visible");
                document.querySelectorAll(`#${{uid}} .pred-cell.selected`).forEach(c => c.classList.remove("selected"));
            }}
        }});

        // Initial build
        const {{ indices }} = computeVisibleLayers(currentCellWidth);
        buildTable(currentCellWidth, indices);
        drawAllTrajectories(null, null, null);
    }})();
    </script>
    '''

    display(HTML(full_html))
