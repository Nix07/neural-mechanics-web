"""
Interactive Logit Lens Visualizations using HTML/CSS

Creates beautiful, interactive heatmaps with hover tooltips showing
top-k predictions at each layer and position.

Usage:
    from logit_lens_viz import logit_lens_heatmap, token_trajectory

    # Heatmap with hover showing top predictions
    logit_lens_heatmap(prompt, model, target_token=" Paris", remote=True)
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
    """Collect logit lens data for all layers and positions."""
    n_layers = model.config.num_hidden_layers
    tokens = model.tokenizer.encode(prompt)
    token_strs = [model.tokenizer.decode([t]) for t in tokens]

    stacked_logits = None
    with model.trace(prompt, remote=remote):
        logits_list = []
        for layer_idx in range(n_layers):
            hidden = model.model.layers[layer_idx].output[0]
            logits = model.lm_head(model.model.norm(hidden))
            seq_logits = logits[0] if len(logits.shape) == 3 else logits
            logits_list.append(seq_logits)
        stacked_logits = torch.stack(logits_list).save()

    stacked_logits = get_value(stacked_logits)
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

    Hover over cells to see top-k predictions at each layer/position.

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

    # Build HTML
    rows_html = []

    for layer_idx in reversed(layer_indices):  # Top to bottom = last to first layer
        cells = []
        for pos in range(seq_len):
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
            tooltip_lines = []
            for p, idx in zip(top_probs, top_indices):
                tok = model.tokenizer.decode([idx])
                tok_escaped = html.escape(repr(tok)[1:-1])
                prob_pct = p.item() * 100
                tooltip_lines.append(f'{tok_escaped}: {prob_pct:.1f}%')

            tooltip = '&#10;'.join(tooltip_lines)  # &#10; is newline in title attr

            cells.append(
                f'<td style="background:{color}; width:{cell_size}px; height:{cell_size}px;" '
                f'title="Layer {layer_idx}, Pos {pos}&#10;{tooltip}"></td>'
            )

        row_html = f'<tr><td class="layer-label">{layer_idx}</td>{"".join(cells)}</tr>'
        rows_html.append(row_html)

    # Token header row
    token_headers = ''.join(
        f'<th class="token-label" title="{html.escape(t)}">{html.escape(t[:4])}</th>'
        for t in token_strs
    )
    header_row = f'<tr><th></th>{token_headers}</tr>'

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
            font-size: 10px;
        }}
        .logit-lens-table td, .logit-lens-table th {{
            padding: 0;
            border: 1px solid #e0e0e0;
        }}
        .logit-lens-table td:hover {{
            outline: 2px solid #333;
            outline-offset: -1px;
            cursor: pointer;
        }}
        .layer-label {{
            padding: 2px 6px !important;
            text-align: right;
            font-weight: 500;
            color: #666;
            background: #f8f8f8;
            border: none !important;
        }}
        .token-label {{
            padding: 4px 2px !important;
            text-align: center;
            font-weight: 500;
            max-width: {cell_size}px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            background: #f8f8f8;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            height: 60px;
        }}
    </style>
    <div class="logit-lens-container">
        <div class="logit-lens-title">{title}</div>
        <div class="logit-lens-subtitle">{subtitle}</div>
        <table class="logit-lens-table">
            {header_row}
            {"".join(rows_html)}
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
