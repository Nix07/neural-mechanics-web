# LogitLensWidget Documentation

An interactive JavaScript widget for visualizing logit lens data from transformer language models. The widget displays top-k token predictions at each layer, with probability trajectories showing how predictions evolve through the network.

## Quick Start

```javascript
// Create widget with logit lens data
var widget = LogitLensWidget("#container", data);

// With custom title
var widget = LogitLensWidget("#container", data, { title: "The quick brown fox" });

// Link two widgets for comparison
widget1.linkColumnsTo(widget2);
```

## Constructor

```javascript
LogitLensWidget(container, data, uiState)
```

### Parameters

#### `container` (string | Element)

Where to render the widget. Accepts:
- CSS selector string: `"#myDiv"`, `".widget-container"`, `"#main .viz:first-child"`
- DOM Element reference: `document.getElementById("myDiv")`

#### `data` (Object)

Logit lens data collected from the model. Structure:

```javascript
{
  layers: [0, 1, 2, ..., 79],           // Layer indices analyzed
  tokens: ["<s>", "The", " quick", ...], // Input token strings
  cells: [                               // 2D array: [position][layer]
    [                                    // Position 0
      {                                  // Layer 0
        token: " the",                   // Top-1 predicted token
        prob: 0.0234,                    // Top-1 probability (0-1)
        trajectory: [0.01, 0.02, ...],   // Top-1's prob at each layer
        topk: [                          // Top-k predictions
          { token: " the", prob: 0.0234, trajectory: [...] },
          { token: " a", prob: 0.0189, trajectory: [...] },
          // ... up to k entries
        ]
      },
      // ... more layers
    ],
    // ... more positions
  ]
}
```

**Data Collection**: Use `collect_logit_lens_topk_efficient()` from `logit_lens_data.py` with `track_across_layers=True` to generate this data structure. The trajectories enable the probability evolution visualization.

#### `uiState` (Object, optional)

Saved UI state to restore. All properties are optional:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `title` | string | "Logit Lens: Top Predictions by Layer" | Widget title (editable) |
| `cellWidth` | number | 44 | Column width in pixels |
| `inputTokenWidth` | number | 100 | Input token column width |
| `chartHeight` | number | 140 | SVG chart height (60-400) |
| `maxRows` | number | null | Max visible rows (null = all) |
| `maxTableWidth` | number | null | Max table width (null = container) |
| `colorMode` | string | "top" | Cell coloring: "top", "none", or token |
| `pinnedGroups` | array | [] | Pinned token groups |
| `pinnedRows` | array | [] | Pinned input positions |
| `colorIndex` | number | 0 | Next color index for groups |
| `lastPinnedGroupIndex` | number | -1 | Last active group |

### Return Value

Returns a widget interface object:

```javascript
{
  uid: "ll_interact_0",              // Unique widget ID
  getState: function() {...},        // Get serializable UI state
  getColumnState: function() {...},  // Get {cellWidth, inputTokenWidth, maxTableWidth}
  setColumnState: function(s) {...}, // Set column sizing
  linkColumnsTo: function(w) {...},  // Link columns to another widget
  unlinkColumns: function(w) {...}   // Unlink columns
}
```

## Interactive Features

### Table Gestures

| Gesture | Target | Effect |
|---------|--------|--------|
| **Click** | Prediction cell | Open popup with top-k predictions |
| **Click** | Input token | Pin/unpin row for trajectory comparison |
| **Click** | Title text | Edit title inline |
| **Click** | "(colored by X)" | Open color mode menu |
| **Hover** | Prediction cell | Show trajectory as gray dotted line |
| **Hover** | Input token row | Highlight row, preview auto-pin token |
| **Drag** | Column border | Resize column width |
| **Drag** | Input column border | Resize input column width |
| **Drag** | Table right edge | Adjust table width constraint |
| **Drag** | Table bottom edge | Truncate visible rows |
| **Drag** | Chart x-axis | Resize chart height |
| **Drag** | Chart y-axis | Resize input column width |

### Popup Interactions

| Gesture | Effect |
|---------|--------|
| **Click** token | Pin/unpin token trajectory (creates new group) |
| **Shift+Click** token | Add/remove token to last active group |
| **Click** X button | Close popup |
| **Click** outside | Close popup |

### Token Pinning

Clicking a token in the popup pins it for persistent trajectory display:
- First pin creates a new colored group
- Shift+click adds tokens to the existing group
- Similar tokens (same after removing punctuation/spaces) show a hint
- Pinned tokens sum their probabilities in the trajectory

### Row Pinning

Clicking an input token pins that row for multi-row comparison:
- Each pinned row uses a different line style (solid, dashed, dotted, dash-dot)
- Auto-pins the highest probability token (>5%) at that position
- Yellow background indicates pinned rows
- Enables comparing how different input positions predict

### Color Modes

Access via the "(colored by X)" button:
- **top prediction**: Cells colored by top-1 probability (default)
- **[specific token]**: Cells colored by that token's probability
- **none**: All cells white (cleaner view)

When the title looks like a prompt (ends with input tokens) and a specific token matching the final prediction is selected, the label simplifies to just show the predicted token.

### Resize Handles

Hover over "showing every N layers..." to reveal all resize handles:
- **Column borders**: Adjust cell width (affects visible layer count)
- **Right edge**: Constrain table width (may introduce layer strides)
- **Bottom edge**: Limit visible rows
- **X-axis**: Adjust chart height (60-400px)
- **Y-axis**: Adjust input column width

## Widget Linking

Link multiple widgets to synchronize column sizing for comparison:

```javascript
var widget1 = LogitLensWidget("#viz1", data1);
var widget2 = LogitLensWidget("#viz2", data2);

// Bidirectional link - resizing either updates both
widget1.linkColumnsTo(widget2);

// Unlink
widget1.unlinkColumns(widget2);
widget2.unlinkColumns(widget1);
```

**Synced properties**: `cellWidth`, `inputTokenWidth`, `maxTableWidth`

**Not synced** (independent per widget): `chartHeight`, `pinnedGroups`, `pinnedRows`, `colorMode`, `title`, `maxRows`

## State Serialization

Save and restore widget state for round-trip:

```javascript
// Get current state
var state = widget.getState();
localStorage.setItem('widgetState', JSON.stringify(state));

// Restore state
var saved = JSON.parse(localStorage.getItem('widgetState'));
var widget = LogitLensWidget("#viz", data, saved);
```

The state object is JSON-serializable and includes all UI configuration.

## Design Rationale

### Layer Stride Display

With many layers (e.g., 80 in Llama 70B), not all can display at once. The widget:
1. Computes how many columns fit given current cell width and container
2. Shows evenly-spaced layers (e.g., "showing every 4 layers")
3. Dragging column borders adjusts the stride dynamically
4. Right-edge dragging constrains max width, affecting stride

### Trajectory Visualization

The probability trajectory shows how a token's prediction strength evolves:
- X-axis: layers (matching table columns)
- Y-axis: probability (0 to dynamic max)
- Solid colored lines: pinned token groups
- Gray dotted line: hovered token (preview)
- Multiple line styles: different pinned rows

### Token Grouping

Similar tokens (e.g., " current" and "current") can be grouped:
- Shift+click adds to the last active group
- Group probabilities sum in the trajectory
- Useful for analyzing case/spacing variants

### Smooth Resizing

Column resize thresholds scale with column count to ensure smooth dragging even with 70+ columns. The threshold is `0.5 / numColumns` pixels.

## CSS Scoping

Each widget instance injects scoped CSS using a unique ID prefix (`#ll_interact_0`, `#ll_interact_1`, etc.), allowing multiple independent widgets on the same page without style conflicts.

## Browser Compatibility

Requires modern browser with:
- CSS `:has()` selector (Chrome 105+, Safari 15.4+, Firefox 121+)
- ES6 template literals
- SVG support

## Files

- `logit_lens_preview.html` - Widget implementation and test page
- `logit_lens_data.py` - Python data collection utilities
- `fetch_preview_data.py` - Script to fetch data from NDIF
- `preview_data.js` - Cached data for testing (JSONP format)
