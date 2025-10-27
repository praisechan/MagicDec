# Latency Breakdown Plotting Script

## Overview
This script (`plot_latency_breakdown.py`) generates a stacked percentage bar chart showing latency breakdown across different batch sizes and configurations.

## Features

### 1. **Multilevel X-axis Ticks**
- **Lower level**: Shows case names (PIM+SD, PIM+SD+LB, PIM+SD+DV)
- **Upper level**: Shows batch sizes (b=8, b=32, b=128)
- Vertical separator lines between batch groups

### 2. **Color Scheme**
Each stage has a distinct color:
- **Draft**: Yellow/Gold (#F4B942)
- **Early-verify**: Blue (#5B9BD5)
- **Final-verify**: Green (#70AD47)

### 3. **Pattern Differentiation**
Within each stage color:
- **GPU**: Solid fill (no pattern)
- **Flash**: Hatched pattern (///) to differentiate from GPU

### 4. **Layout Structure**
- Each batch size contains 3 bars (one per case)
- Each bar combines both GPU and Flash components stacked together
- Within each stage color, GPU (solid) is stacked first, then Flash (hatched)
- Bars show stacked percentages totaling 100% (combining all stages and compute types)
- Legend shows all 6 combinations (3 stages × 2 compute types)
- Vertical divider lines separate batch groups (manually controllable)

## Usage

```bash
cd /home/juchanlee/MagicDec/figure/latency_breakdown
python plot_latency_breakdown.py
```

## Output Files
- `latency_breakdown_qwen14b_16K.png` - High-resolution PNG (300 DPI)
- `latency_breakdown_qwen14b_16K.pdf` - Vector PDF for publications

## Input Data Format
The script reads `simulation_latency_breakdown_qwen14b_16K_new.CSV` with the following structure:

```
,,Draft,,Early-verify,,Final-verify,
,,GPU,Flash,GPU,Flash,GPU,Flash
b=8,PIM+SD,30.38,25.31,7.67,29.50,1.08,6.06
,PIM+SD+LB,30.38,12.06,7.67,22.23,1.08,6.06
,PIM+SD+DV,26.53,10.53,6.69,16.21,0.96,5.35
```

- First 2 rows are headers
- Column 0: Batch size (appears only once per group)
- Column 1: Case name
- Columns 2-7: Latency values for each stage/compute type pair

## Customization

### Colors
Modify the `stage_colors` dictionary:
```python
stage_colors = {
    'Draft': '#F4B942',
    'Early-verify': '#5B9BD5',
    'Final-verify': '#70AD47'
}
```

### Figure Size
Change in the `plt.subplots()` call:
```python
fig, ax = plt.subplots(figsize=(12, 6))  # width, height in inches
```

### Bar Width and Spacing
Adjust these variables:
```python
bar_width = 0.6      # Width of each bar
bar_gap = 1.0        # Gap between different cases within a batch
group_gap = 2.0      # Gap between batch size groups
```

### Manual Divider Line Control
To manually set the positions of vertical divider lines between batch groups, uncomment and modify this line in the script:
```python
# CUSTOMIZE DIVIDER LINES HERE:
# Uncomment and modify these lines to manually set divider positions
# line_positions = [-1.0, 4.5, 9.5, 14.0]  # Example: custom positions
```

The `line_positions` array should have `n_batches + 1` elements (4 elements for 3 batch sizes):
- First element: Left edge (before b=8)
- Middle elements: Dividers between batch groups
- Last element: Right edge (after b=128)

### Patterns
Modify the hatch pattern for Flash:
```python
hatch = '///' if compute_type == 'Flash' else None
# Options: '///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo', 'OOO', '...', '**'
```
