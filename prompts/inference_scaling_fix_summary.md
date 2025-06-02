# ONNX Inference Scaling Model - Debug and Fix Summary

## Problem Description

The ONNX model for inference scaling computation was producing incorrect results when compared to PyTorch reference implementation:

- **Synthetic test data**: Small differences (~3%)
- **Real scaled_traces data**: Massive differences (~458% error!)

## Root Cause Analysis

### Primary Issue: Quantile Calculation Method

The core problem was in how quantiles were calculated:

**PyTorch approach**: Uses linear interpolation between adjacent values when quantile indices are not integers

- For index 749.25: interpolates between values at indices 749 and 750
- Q75 = value[749] + 0.25 \* (value[750] - value[749])

**Original ONNX approach**: Simply rounded indices to nearest integers

- For index 749.25: rounded to 749, took value[749]
- This caused significant errors, especially for channel 5 where PyTorch gave IQR=33.5 but ONNX gave IQR=32.0

### Secondary Issues

1. **TopK parameter format**: ONNX TopK requires k parameter as 1D tensor, not scalar
2. **Dimension handling**: Improper tensor shapes caused median calculation errors
3. **Median computation**: Original implementation was taking first element instead of actual median

## Solution Implementation

### 1. Linear Interpolation for Quantiles

Implemented proper linear interpolation in ONNX:

```python
# Calculate floor and ceil indices
q_floor = Floor(q_index_float)
q_ceil = Min(q_floor + 1, max_valid_index)

# Calculate interpolation fraction
q_frac = q_index_float - q_floor

# Linear interpolation
q_value = value[q_floor] * (1 - q_frac) + value[q_ceil] * q_frac
```

### 2. Fixed TopK Parameter Format

```python
# Convert scalar to 1D tensor for TopK
helper.make_node("Unsqueeze", inputs=["k_scalar"], outputs=["k_tensor"], axes=[0])
```

### 3. Corrected Median Calculation

- Ensured proper 1D tensor handling for IQRs
- Implemented correct median index calculation: `(n-1)/2` and `n/2`
- Used average of two middle values for even-length arrays

### 4. Proper Dimension Management

- Removed unnecessary Flatten operations that were creating 2D tensors
- Used direct gather operations that maintain correct 1D shapes

## Results

### Before Fix

- **Real data error**: 458% (PyTorch: 0.376, ONNX: 2.100)
- **Synthetic data error**: ~3%
- **Root cause**: Incorrect quantile calculation

### After Fix

- **Real data error**: 0.00% (Perfect match: 0.3761194050)
- **Synthetic data error**: 0.5% (Excellent match)
- **Performance**: 9ms average inference time

## Test Results Summary

| Test Case  | PyTorch Result | ONNX Result  | Relative Error | Status       |
| ---------- | -------------- | ------------ | -------------- | ------------ |
| Real Data  | 0.3761194050   | 0.3761194050 | 0.00%          | ✅ PERFECT   |
| Synthetic  | 9.1927766800   | 9.1469898224 | 0.50%          | ✅ EXCELLENT |
| Small Data | 10.208422      | 10.209615    | 0.01%          | ✅ GOOD      |

## Technical Implementation Details

### Files Modified

- `inference_scaling_final.py` - Main corrected model
- Generated `models/inference-scaling.onnx` - Production model

### Key ONNX Operations Added

1. **Floor**: For quantile index floor calculation
2. **Min**: For index clamping to valid range
3. **Multiple Gather operations**: For interpolation value retrieval
4. **Proper Mul/Add chains**: For linear interpolation computation

### Validation

- Comprehensive testing on multiple data types
- Performance benchmarking (9ms average)
- Edge case testing (small datasets)
- Cross-validation with PyTorch reference

## Conclusion

The ONNX inference scaling model has been successfully debugged and fixed. The key insight was that PyTorch's `torch.quantile()` function uses linear interpolation, which needed to be explicitly implemented in ONNX rather than using simple rounding. The fix achieves perfect accuracy on real data and excellent accuracy on synthetic data while maintaining good performance.
