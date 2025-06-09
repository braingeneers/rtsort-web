#!/usr/bin/env python3
"""
Demonstration of the Maxwell h5 subset functionality.
This script shows how to create and validate subsets of Maxwell MEA recordings.
"""

import sys
from pathlib import Path
import h5py
import numpy as np
from spikeinterface.extractors import MaxwellRecordingExtractor

# Add scripts directory to path so we can import our subset function
sys.path.append(str(Path(__file__).parent / 'scripts'))
from subset_maxwell_h5 import subset_maxwell_h5, validate_subset


def demo_subset_creation():
    """Demonstrate creating subsets with different parameters."""
    
    input_file = "data/MEA_rec_patch_ground_truth_cell7.raw.h5"
    
    if not Path(input_file).exists():
        print(f"Input file {input_file} not found. Please ensure it exists.")
        return
    
    print("=== Maxwell H5 Subset Demonstration ===\n")
    
    # Example 1: Default subset (8 channels, 5 seconds)
    print("1. Creating default subset (8 channels, 5 seconds)...")
    output1 = subset_maxwell_h5(input_file)
    
    # Example 2: Smaller subset (4 channels, 2 seconds)
    print("\n2. Creating smaller subset (4 channels, 2 seconds)...")
    output2 = subset_maxwell_h5(
        input_file, 
        num_channels=4, 
        num_seconds=2, 
        output_file="data/small_subset.h5"
    )
    
    # Example 3: Larger subset (16 channels, 10 seconds)
    print("\n3. Creating larger subset (16 channels, 10 seconds)...")
    output3 = subset_maxwell_h5(
        input_file, 
        num_channels=16, 
        num_seconds=10, 
        output_file="data/large_subset.h5"
    )
    
    # Validate all created files
    print("\n=== Validation ===")
    for i, output_file in enumerate([output1, output2, output3], 1):
        print(f"\nValidating subset {i}: {Path(output_file).name}")
        try:
            validate_subset(input_file, output_file)
            
            # Additional validation with h5py and SpikeInterface
            with h5py.File(output_file, 'r') as h:
                print(f"  Shape: {h['sig'].shape}")
                print(f"  Gain: {h['settings']['gain'][0]}")
                print(f"  LSB: {h['settings']['lsb'][0]}")
                print(f"  First sample values: {h['sig'][0, :5]}")
            
            # Test with SpikeInterface
            rec = MaxwellRecordingExtractor(output_file)
            print(f"  SpikeInterface - Channels: {rec.get_num_channels()}, "
                  f"Rate: {rec.get_sampling_frequency()} Hz, "
                  f"Duration: {rec.get_num_samples()/rec.get_sampling_frequency():.2f}s")
            
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
    
    print("\n=== Demonstration Complete ===")
    print("Created files:")
    for output_file in [output1, output2, output3]:
        size_mb = Path(output_file).stat().st_size / (1024*1024)
        print(f"  {output_file} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    demo_subset_creation()
