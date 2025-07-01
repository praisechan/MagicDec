# Dynamic Budget Adjustment for 3-Stage Hierarchical Speculative Decoding

This enhanced implementation adds dynamic budget adjustment to the 3-stage hierarchical speculative decoding system based on the confidence scores from the draft stage.

## New Features

### Dynamic Budget Adjustment
- The system now monitors the `top1_top2_diff` values from the draft speculation stage
- When all tokens have high confidence (above the threshold), it uses a lower budget for verification
- This can improve efficiency by reducing computation when the draft model is confident about its predictions

### New Command Line Arguments

- `--budget2_low`: Lower budget ratio for verification when confidence is high (default: 0.1)
- `--confidence_threshold`: Threshold for top1_top2_diff to trigger lower budget usage (default: 0.5)

### How It Works

1. **Draft Stage**: `engine.speculate()` generates tokens and returns confidence scores (`top1_top2_diff`)
2. **Confidence Analysis**: The system checks if all tokens have confidence above the threshold
3. **Dynamic Budget**: 
   - High confidence → Use `budget2_low` for verification
   - Low confidence → Use original `budget2` for verification
4. **Verification Stage**: `engine.verify()` uses the dynamically adjusted budget
5. **Settlement Stage**: `engine.settle()` performs final verification with full attention

### Example Usage

```bash
python run_3step.py \
    --model_name "llama-3.1-8b" \
    --dataset "pg19" \
    --budget1 0.05 \
    --budget2 0.25 \
    --budget2_low 0.1 \
    --confidence_threshold 0.5 \
    --attn_type "RetroInfer" \
    --printoutput
```

### Output Statistics

The system now provides detailed statistics including:
- Total verification and settlement steps
- Number of budget switches
- Confidence threshold used
- Budget ratios (original vs. low confidence)

### Benefits

- **Adaptive Efficiency**: Uses lower computational budget when the model is confident
- **Quality Preservation**: Maintains full budget when uncertainty is detected
- **Transparency**: Provides detailed logging of budget decisions
- **Configurability**: Allows tuning of confidence thresholds and budget ratios

## Implementation Details

### Backend Changes (`backend_for_3stage.py`)
- Added `update_verification_budget()` method to dynamically adjust attention configuration

### Main Loop Changes (`run_3step.py`)
- Modified speculation call to capture confidence scores
- Added dynamic budget logic based on confidence analysis
- Enhanced logging and statistics tracking

The system maintains backward compatibility - if confidence data isn't available, it falls back to the original budget strategy.
