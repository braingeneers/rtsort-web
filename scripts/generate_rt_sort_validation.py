#!/usr/bin/env python3
"""
Generate numerical validation reference data for rt_sort.py processing.

This script processes an H5 file through the same pipeline as run_detection_model
but outputs a comprehensive JSON file with all intermediate and final values
for validation against the TypeScript worker implementation.
"""

import json
import sys
from pathlib import Path
import numpy as np
import scipy.stats
import torch
import h5py
from spikeinterface.extractors import MaxwellRecordingExtractor

# Import the model from the braindance package
sys.path.append("/Users/rcurrie/braindance")
from braindance.core.spikedetector.model import ModelSpikeSorter

# Number of validation values to compare
N_VALIDATION_VALUES = 10


def extract_h5_parameters(h5_file_path):
    """Extract parameters from Maxwell H5 file using both h5py and SpikeInterface."""

    # Use SpikeInterface for comprehensive recording info
    recording = MaxwellRecordingExtractor(h5_file_path)

    # Use h5py for direct access to settings
    with h5py.File(h5_file_path, "r") as h5:
        settings = h5["settings"]
        gain = float(settings["gain"][0])
        lsb = float(settings["lsb"][0])

        # Try to get HPF if available
        try:
            hpf = float(settings["hpf"][0])
        except:
            hpf = None

    return {
        "h5_file_path": str(h5_file_path),
        "num_channels": recording.get_num_channels(),
        "num_samples": recording.get_num_samples(),
        "sampling_rate": recording.get_sampling_frequency(),
        "duration": recording.get_total_duration(),
        "gain": gain,
        "lsb": lsb,
        "hpf": hpf,
        "has_scaled_traces": recording.has_scaleable_traces(),
        "channel_gains": (
            recording.get_channel_gains().tolist()
            if recording.has_scaleable_traces()
            else None
        ),
    }


def load_and_scale_raw_data(
    h5_file_path, num_channels, num_samples_to_read=2000, start_sample=0
):
    """Load raw data from H5 file and convert to scaled traces (microvolts) using the same method as detect_sequences (SpikeInterface method)."""

    # Use SpikeInterface exactly as detect_sequences would
    recording = MaxwellRecordingExtractor(h5_file_path)

    # Get traces using SpikeInterface
    traces = recording.get_traces(
        start_frame=start_sample, end_frame=start_sample + num_samples_to_read
    )

    # Apply channel gains manually (same as detect_sequences when has_scaleable_traces() is True)
    if recording.has_scaleable_traces():
        gains = recording.get_channel_gains()
        traces = traces * gains

    # Convert to the format that rt_sort.py expects: (channels, samples)
    traces = traces.T

    # Convert to float16 as rt_sort.py does
    scaled_traces = traces.astype(np.float16)

    return scaled_traces  # Shape: (num_channels, num_samples)


def calculate_inference_scaling(
    scaled_traces, inference_scaling_numerator=12.6, pre_median_frames=1000
):
    """Calculate inference scaling using the exact logic from rt_sort.py."""

    # Extract window for inference scaling calculation
    window = scaled_traces[:, :pre_median_frames]

    # Calculate IQRs using scipy.stats.iqr (same as rt_sort.py)
    iqrs = scipy.stats.iqr(window, axis=1)
    median_iqr = np.median(iqrs)

    # Calculate inference scaling
    inference_scaling = (
        inference_scaling_numerator / median_iqr if median_iqr != 0 else 1
    )

    return inference_scaling, median_iqr, iqrs


def run_pytorch_model(
    scaled_traces,
    model,
    inference_scaling,
    sample_size=200,
    num_output_locs=120,
    device="cpu",
):
    """Run PyTorch model on scaled traces using exact logic from rt_sort.py."""

    input_scale = model.input_scale

    # Process first window (frames 0 to sample_size)
    start_frame = 0
    traces_torch = torch.tensor(
        scaled_traces[:, start_frame : start_frame + sample_size],
        device=device,
        dtype=torch.float16,
    )

    # Subtract median for baseline correction (exact rt_sort.py logic)
    traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values

    # Prepare input for model
    model_input = traces_torch[:, None, :] * input_scale * inference_scaling

    # Run model
    with torch.no_grad():
        outputs = model.model(model_input).cpu()

    print(f"Debug: Model output shape: {outputs.shape}")

    # Handle different output shapes
    if outputs.dim() == 3:
        # Expected shape: (num_channels, 1, num_output_locs)
        model_output = outputs[:, 0, :].numpy()
    else:
        # Shape: (num_channels, num_output_locs)
        model_output = outputs.numpy()

    return {
        "raw_window": scaled_traces[
            :, start_frame : start_frame + sample_size
        ].tolist(),
        "median_subtracted": traces_torch.cpu().numpy().tolist(),
        "model_input": model_input.cpu().numpy().tolist(),
        "model_output": model_output.tolist(),
    }


def generate_validation_data(
    h5_file_path, model_path, output_json_path, n_values=N_VALIDATION_VALUES
):
    """Generate comprehensive validation data."""

    print(f"Processing {h5_file_path}...")

    # 1. Extract H5 file parameters
    print("Extracting H5 file parameters...")
    h5_params = extract_h5_parameters(h5_file_path)

    # 2. Load and scale raw data
    print("Loading and scaling raw data...")
    scaled_traces = load_and_scale_raw_data(
        h5_file_path,
        h5_params["num_channels"],
        num_samples_to_read=2000,  # Enough for inference scaling + model processing
    )

    # 3. Load PyTorch model
    print("Loading PyTorch model...")
    model = ModelSpikeSorter.load(model_path)
    model.model.eval()

    # 4. Calculate inference scaling
    print("Calculating inference scaling...")
    inference_scaling_numerator = 12.6
    pre_median_frames = 1000

    inference_scaling, median_iqr, iqrs = calculate_inference_scaling(
        scaled_traces, inference_scaling_numerator, pre_median_frames
    )

    # 5. Run PyTorch model
    print("Running PyTorch model...")
    model_results = run_pytorch_model(
        scaled_traces,
        model,
        inference_scaling,
        sample_size=model.sample_size,
        num_output_locs=model.num_output_locs,
    )

    # 6. Compile validation data
    validation_data = {
        "metadata": {
            "h5_file": str(h5_file_path),
            "model_path": str(model_path),
            "generated_by": "generate_rt_sort_validation.py",
            "description": "Reference numerical values for rt_sort.py validation",
        },
        # Constants from Python code
        "constants": {
            "inference_scaling_numerator": inference_scaling_numerator,
            "pre_median_frames": pre_median_frames,
            "sample_size": model.sample_size,
            "num_output_locs": model.num_output_locs,
            "input_scale": float(model.input_scale),
            "buffer_front_sample": model.buffer_front_sample,
            "buffer_end_sample": model.buffer_end_sample,
            "n_validation_values": n_values,
        },
        # Values extracted from H5 file
        "h5_file_parameters": h5_params,
        # Computed values
        "computed_values": {
            "inference_scaling": float(inference_scaling),
            "median_iqr": float(median_iqr),
            "iqrs_first_n": iqrs[:n_values].tolist(),
            "iqrs_stats": {
                "mean": float(np.mean(iqrs)),
                "std": float(np.std(iqrs)),
                "min": float(np.min(iqrs)),
                "max": float(np.max(iqrs)),
            },
        },
        # Reduced sample numerical values for validation (only 4 key data types)
        "sample_values": {
            # 1) Raw values from h5 from channel 0 (scaled traces are the "raw" values we use)
            "raw_channel_0": scaled_traces[0, :n_values].tolist(),
            # 2) Scaled values from channel 0 (same as raw in our case since we use SpikeInterface scaling)
            "scaled_channel_0": scaled_traces[0, :n_values].tolist(),
            # 3) Model input values from channel 0
            "model_input_channel_0": model_results["model_input"][0][0][:n_values],
            # 4) Model output values from channel 0
            "model_output_channel_0": model_results["model_output"][0][:n_values],
        },
        # Full model processing results for first window
        "model_processing": {
            "window_start_frame": 0,
            "window_end_frame": model.sample_size,
            "num_channels": h5_params["num_channels"],
            "window_data": model_results["raw_window"][
                :2
            ],  # Only first 2 channels for debugging
        },
    }

    # 7. Save to JSON
    print(f"Saving validation data to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump(validation_data, f, indent=2)

    print("Validation data generation complete!")
    print(f"Key values:")
    print(f"  - Inference scaling: {inference_scaling}")
    print(f"  - Median IQR: {median_iqr}")
    print(f"  - First raw value (channel 0): {scaled_traces[0, 0]}")
    print(f"  - First model output (channel 0): {model_results['model_output'][0][0]}")

    return validation_data


def main():
    """Main function."""

    # Default paths - can be modified as needed
    h5_file_path = Path("public/test_maxwell_raw.h5")
    model_path = Path("checkpoints/spikedetector/mea")
    output_json_path = Path("public/models/test_maxwell_raw.validation.json")

    # Check if files exist
    if not h5_file_path.exists():
        print(f"Error: H5 file not found: {h5_file_path}")
        print("Please ensure the H5 file exists or modify the path in the script.")
        return 1

    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        print("Please ensure the model exists or modify the path in the script.")
        return 1

    # Create output directory
    output_json_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        validation_data = generate_validation_data(
            h5_file_path, model_path, output_json_path, n_values=N_VALIDATION_VALUES
        )
        return 0

    except Exception as e:
        print(f"Error generating validation data: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
