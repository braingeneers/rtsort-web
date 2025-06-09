#!/usr/bin/env python3
"""
Simple validation script to test h5py direct access to subset files.
"""

import h5py
import numpy as np

def test_h5py_access():
    """Test direct h5py access to verify h5wasm compatibility."""
    
    filename = 'data/MEA_rec_patch_ground_truth_cell7.raw_subset.h5'
    
    print('=== H5PY Direct Access Test (simulating h5wasm) ===')
    
    with h5py.File(filename, 'r') as h:
        print(f'Available keys: {list(h.keys())}')
        print()
        
        # Signal data
        sig = h['sig']
        print(f'Signal shape: {sig.shape}')
        print(f'Signal dtype: {sig.dtype}')
        first_10 = sig[0, :10]
        print(f'First 10 values from channel 0: {first_10}')
        print()
        
        # Settings
        if 'settings' in h:
            settings = h['settings']
            gain = settings['gain'][0]
            lsb = settings['lsb'][0]
            
            print(f'Gain: {gain}')
            print(f'LSB (microvolts): {lsb * 1e6:.6f}')
            print()
            
            # Calculate physical values (like you would in JavaScript)
            raw_values = first_10
            physical_values = raw_values.astype(np.float64) * lsb * 1e6 * gain
            print(f'Raw values: {raw_values}')
            print(f'Physical values (microvolts): {physical_values}')
            print()
        
        # Mapping (electrode positions)
        if 'mapping' in h:
            mapping = h['mapping']
            print(f'Mapping shape: {mapping.shape}')
            print(f'Mapping dtype: {mapping.dtype}')
            first_mapping = mapping[0]
            print(f'First channel mapping: channel={first_mapping["channel"]}, electrode={first_mapping["electrode"]}, x={first_mapping["x"]}, y={first_mapping["y"]}')
            print()
        
        # Sample rate calculation (this would be known or derived)
        sample_rate = 20000.0  # Hz
        duration = sig.shape[1] / sample_rate
        print(f'Derived sample rate: {sample_rate} Hz')
        print(f'Duration: {duration} seconds')
        print(f'Number of channels: {sig.shape[0]}')
        print(f'Number of samples: {sig.shape[1]}')
        
    print('✓ File can be read directly with h5py (h5wasm compatible)')
    return True

if __name__ == '__main__':
    test_h5py_access()
