# Inference Scaling Calculation Fix Summary

## Problem
The `calculateInferenceScaling` function in `worker.ts` was not producing results within 4 decimal places of the expected value `0.3761194029850746`. The original TypeScript implementation was using simplified quantile and median calculations that didn't match PyTorch's behavior.

## Root Causes

### 1. Quantile Calculation Method
**Original (Incorrect):**
```typescript
const q25Index = Math.floor(0.25 * (preMedianFrames - 1))
const q75Index = Math.floor(0.75 * (preMedianFrames - 1))
const q25 = channelData[q25Index]
const q75 = channelData[q75Index]
```

**Fixed (PyTorch-compatible):**
```typescript
const q25 = linearInterpolate(channelData, 0.25)
const q75 = linearInterpolate(channelData, 0.75)
```

The original used simple floor indexing, but PyTorch's `torch.quantile` uses linear interpolation between data points.

### 2. Median Calculation Method
**Original (Incorrect):**
```typescript
const sortedIqrs = Array.from(iqrs).sort((a, b) => a - b)
const medianIqr = sortedIqrs[Math.floor(sortedIqrs.length / 2)]
```

**Fixed (PyTorch-compatible):**
```typescript
function calculateMedian(data: Float32Array): number {
  const sorted = new Float32Array(data)
  sorted.sort()
  
  const n = sorted.length
  if (n % 2 === 0) {
    // For even length, return average of two middle elements
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
  } else {
    // For odd length, return middle element
    return sorted[Math.floor(n / 2)]
  }
}
```

The original only used the middle element for even-length arrays, but PyTorch's `torch.median` averages the two middle elements for even-length arrays.

## Solution

### Created Two Standalone Scripts

1. **`calculate_inference_scaling.py`** - Validates the exact PyTorch computation
   - Loads `validation/scaled_traces.npy` or `public/models/scaled_traces.npy`
   - Uses `torch.quantile` and `torch.median` exactly as in the original notebook
   - Produces: `0.3761194029850746` (exact match)

2. **`calculate_inference_scaling.ts`** - TypeScript implementation with correct algorithms
   - Loads `public/models/scaled_traces.bin` (float16 binary data)
   - Implements linear interpolation for quantiles (matching PyTorch)
   - Implements proper median calculation for even-length arrays
   - Produces: `0.3761194029850746` (exact match)

### Updated worker.ts

- Replaced quantile calculation with `linearInterpolate()` function
- Replaced median calculation with `calculateMedian()` function
- Removed hardcoded inference scaling override
- Added proper helper functions for PyTorch-compatible calculations

## Verification

Both standalone scripts now produce the exact expected value:
- **Python**: `0.3761194029850746` ✅
- **TypeScript**: `0.3761194029850746` ✅
- **Difference**: `0` (exact match within floating-point precision)

The key insight was that PyTorch's statistical functions use more sophisticated algorithms than simple array indexing, particularly for quantile calculation with linear interpolation and median calculation for even-length arrays.
