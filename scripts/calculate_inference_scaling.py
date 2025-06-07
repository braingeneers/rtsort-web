#!/usr/bin/env python3
"""
Standalone script to calculate inference scaling from scaled_traces.npy
This should produce the exact value: 0.3761194029850746
"""

import numpy as np
import torch

def calculate_inference_scaling_torch(
    scaled_traces_path: str = "validation/scaled_traces.npy",
    inference_scaling_numerator: float = 12.6,
    pre_median_frames: int = 1000
) -> float:
    """
    Calculate inference scaling using the exact same logic as in the jupyter notebook.
    
    Args:
        scaled_traces_path: Path to the scaled_traces.npy file
        inference_scaling_numerator: Numerator for scaling calculation (12.6)
        pre_median_frames: Number of frames to use for initial window (1000)
    
    Returns:
        The computed inference scaling value
    """
    
    # Load scaled traces
    print(f"Loading scaled traces from: {scaled_traces_path}")
    scaled_traces = np.load(scaled_traces_path)
    print(f"Scaled traces shape: {scaled_traces.shape}")
    
    # Convert to torch tensor with float16 then cast to float32 for computation
    # This matches exactly what the notebook does
    scaled_traces = torch.tensor(scaled_traces, dtype=torch.float16)
    
    # Extract initial window and cast to float32 for computation
    window = scaled_traces[:, :pre_median_frames].to(torch.float32).cpu()
    
    print(f"Window shape: {window.shape}")
    print(f"Window dtype: {window.dtype}")
    
    if window.dtype != torch.float32:
        raise ValueError(f"Window tensor dtype is {window.dtype}, expected torch.float32")
    
    # Calculate IQRs using torch.quantile (this is the exact computation from the notebook)
    iqrs = torch.quantile(window, 0.75, dim=1) - torch.quantile(window, 0.25, dim=1)
    print(f"IQRs shape: {iqrs.shape}")
    print(f"IQRs: {iqrs}")
    
    # Calculate median of IQRs
    median_iqr = torch.median(iqrs)
    print(f"Median IQR: {median_iqr.item()}")
    
    # Calculate inference scaling
    inference_scaling = inference_scaling_numerator / median_iqr if median_iqr != 0 else 1
    
    print(f"Inference scaling numerator: {inference_scaling_numerator}")
    print(f"Inference scaling: {inference_scaling.item()}")
    
    return inference_scaling.item()

def main():
    """Main function to test the calculation"""
    
    print("=" * 60)
    print("INFERENCE SCALING CALCULATION - PYTHON")
    print("=" * 60)
    
    # Test with validation data
    try:
        result = calculate_inference_scaling_torch("validation/scaled_traces.npy")
        print(f"\n✅ RESULT: {result}")
        
        # Check if it matches the expected value
        expected = 0.3761194029850746
        print(f"Expected: {expected}")
        print(f"Match (4 decimal places): {abs(result - expected) < 1e-4}")
        print(f"Difference: {abs(result - expected)}")
        
    except FileNotFoundError:
        print("❌ validation/scaled_traces.npy not found")
    
    # Also test with public models data if available
    try:
        print("\n" + "-" * 40)
        print("Testing with public/models/scaled_traces.npy")
        result2 = calculate_inference_scaling_torch("public/models/scaled_traces.npy")
        print(f"✅ RESULT: {result2}")
        
        expected = 0.3761194029850746
        print(f"Expected: {expected}")
        print(f"Match (4 decimal places): {abs(result2 - expected) < 1e-4}")
        print(f"Difference: {abs(result2 - expected)}")
        
    except FileNotFoundError:
        print("❌ public/models/scaled_traces.npy not found")

if __name__ == "__main__":
    main()
