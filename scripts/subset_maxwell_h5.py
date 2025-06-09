#!/usr/bin/env python3
"""
Script to create a subset of a Maxwell MEA h5 file with fewer channels and shorter duration.

This script takes a Maxwell raw h5 file and creates a new file with only the first N channels
and first M seconds of data, preserving the essential structure needed for analysis.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from spikeinterface.extractors import MaxwellRecordingExtractor


def subset_maxwell_h5(input_file, num_channels=8, num_seconds=5, output_file=None):
    """
    Create a subset of a Maxwell h5 file with fewer channels and shorter duration.
    
    Parameters:
    -----------
    input_file : str or Path
        Path to the input Maxwell h5 file
    num_channels : int, default 8
        Number of channels to include (starting from channel 0)
    num_seconds : float, default 5
        Number of seconds of data to include
    output_file : str or Path, optional
        Output file path. If None, defaults to input file stem + 'subset.h5'
    
    Returns:
    --------
    str : Path to the created output file
    """
    input_path = Path(input_file)
    if output_file is None:
        output_file = input_path.parent / (input_path.stem + '_subset.h5')
    else:
        output_file = Path(output_file)
    
    print(f"Creating subset from {input_file}")
    print(f"Output file: {output_file}")
    print(f"Channels: {num_channels}, Duration: {num_seconds} seconds")
    
    # First, get recording info using SpikeInterface
    try:
        recording = MaxwellRecordingExtractor(str(input_path))
        sampling_rate = recording.get_sampling_frequency()
        total_channels = recording.get_num_channels()
        total_samples = recording.get_num_samples()
        
        print(f"Original file: {total_channels} channels, {total_samples} samples, {sampling_rate} Hz")
        
        # Calculate number of samples for the requested duration
        num_samples = int(num_seconds * sampling_rate)
        if num_samples > total_samples:
            num_samples = total_samples
            print(f"Warning: Requested duration longer than available data. Using {total_samples/sampling_rate:.2f} seconds")
        
        if num_channels > total_channels:
            num_channels = total_channels
            print(f"Warning: Requested channels exceed available channels. Using {total_channels} channels")
            
    except Exception as e:
        print(f"Error reading with SpikeInterface: {e}")
        print("Falling back to direct h5py reading...")
        
        # Fallback to h5py if SpikeInterface fails
        with h5py.File(input_path, 'r') as h:
            sig_shape = h['sig'].shape
            total_channels = sig_shape[0]
            total_samples = sig_shape[1]
            # Assume 20kHz sampling rate as default for Maxwell
            sampling_rate = 20000.0
            
            num_samples = int(num_seconds * sampling_rate)
            if num_samples > total_samples:
                num_samples = total_samples
            if num_channels > total_channels:
                num_channels = total_channels
    
    print(f"Subset will have: {num_channels} channels, {num_samples} samples")
    
    # Create the subset file
    with h5py.File(input_path, 'r') as input_h5:
        with h5py.File(output_file, 'w') as output_h5:
            
            # Copy essential datasets and groups
            
            # 1. Copy version if it exists
            if 'version' in input_h5:
                output_h5.create_dataset('version', data=input_h5['version'][()])
            
            # 2. Copy settings group (gain, lsb, hpf)
            if 'settings' in input_h5:
                settings_grp = output_h5.create_group('settings')
                for key in input_h5['settings'].keys():
                    settings_grp.create_dataset(key, data=input_h5['settings'][key][()])
            
            # 3. Create subset signal data
            print("Copying signal data...")
            subset_sig = input_h5['sig'][:num_channels, :num_samples]
            output_h5.create_dataset('sig', data=subset_sig, dtype=np.uint16)
            
            # 4. Create subset mapping (if it exists)
            if 'mapping' in input_h5:
                mapping_data = input_h5['mapping'][:]
                # Keep only mappings for the channels we're including
                subset_mapping = mapping_data[:num_channels]
                output_h5.create_dataset('mapping', data=subset_mapping)
            
            # 5. Copy time data (if it exists) - subset to match samples
            if 'time' in input_h5:
                time_data = input_h5['time'][:num_samples]
                output_h5.create_dataset('time', data=time_data)
            
            # 6. Copy other potentially useful datasets (but not groups)
            for key in ['bits', 'message_0']:
                if key in input_h5:
                    try:
                        if isinstance(input_h5[key], h5py.Dataset):
                            output_h5.create_dataset(key, data=input_h5[key][()])
                        else:
                            print(f"Skipping {key} (not a dataset)")
                    except Exception as e:
                        print(f"Warning: Could not copy {key}: {e}")
    
    print(f"Subset file created: {output_file}")
    return str(output_file)


def validate_subset(original_file, subset_file):
    """
    Validate that the subset file was created correctly by comparing key parameters
    and first few samples with the original.
    """
    print("\nValidating subset file...")
    
    with h5py.File(original_file, 'r') as orig_h5:
        with h5py.File(subset_file, 'r') as subset_h5:
            
            # Check basic structure
            print(f"Original signal shape: {orig_h5['sig'].shape}")
            print(f"Subset signal shape: {subset_h5['sig'].shape}")
            
            # Check settings match
            if 'settings' in orig_h5 and 'settings' in subset_h5:
                orig_gain = orig_h5['settings']['gain'][0]
                subset_gain = subset_h5['settings']['gain'][0]
                orig_lsb = orig_h5['settings']['lsb'][0]
                subset_lsb = subset_h5['settings']['lsb'][0]
                
                print(f"Original gain: {orig_gain}, subset gain: {subset_gain}")
                print(f"Original LSB: {orig_lsb}, subset LSB: {subset_lsb}")
                
                assert np.isclose(orig_gain, subset_gain), "Gain values don't match"
                assert np.isclose(orig_lsb, subset_lsb), "LSB values don't match"
            
            # Check first 10 samples of first channel match
            orig_samples = orig_h5['sig'][0, :10]
            subset_samples = subset_h5['sig'][0, :10]
            
            print(f"Original first 10 samples (channel 0): {orig_samples}")
            print(f"Subset first 10 samples (channel 0): {subset_samples}")
            
            assert np.array_equal(orig_samples, subset_samples), "First samples don't match"
            
            # Try to load with SpikeInterface
            try:
                subset_recording = MaxwellRecordingExtractor(subset_file)
                print(f"SpikeInterface validation - Channels: {subset_recording.get_num_channels()}")
                print(f"SpikeInterface validation - Sampling rate: {subset_recording.get_sampling_frequency()}")
                print(f"SpikeInterface validation - Samples: {subset_recording.get_num_samples()}")
                print("✓ Subset file can be opened with SpikeInterface")
            except Exception as e:
                print(f"✗ Warning: Subset file cannot be opened with SpikeInterface: {e}")
    
    print("✓ Validation completed successfully")


def main():
    parser = argparse.ArgumentParser(description='Create a subset of a Maxwell MEA h5 file')
    parser.add_argument('input_file', help='Path to input Maxwell h5 file')
    parser.add_argument('-c', '--channels', type=int, default=8, 
                        help='Number of channels to include (default: 8)')
    parser.add_argument('-s', '--seconds', type=float, default=5, 
                        help='Number of seconds to include (default: 5)')
    parser.add_argument('-o', '--output', help='Output file path (default: input_stem_subset.h5)')
    parser.add_argument('--validate', action='store_true', 
                        help='Validate the subset file after creation')
    
    args = parser.parse_args()
    
    # Create subset
    output_file = subset_maxwell_h5(
        args.input_file, 
        num_channels=args.channels, 
        num_seconds=args.seconds, 
        output_file=args.output
    )
    
    # Validate if requested
    if args.validate:
        validate_subset(args.input_file, output_file)


if __name__ == '__main__':
    main()
