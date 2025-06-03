#!/usr/bin/env python3
"""
Export numpy arrays to binary format for TypeScript consumption.

This script exports:
- scaled_traces.npy -> scaled_traces.bin (float16 data)
- model_outputs.npy -> model_outputs.bin (float16 data)

Along with metadata JSON files containing shape and data type information.
"""

import numpy as np
import json
import argparse
from pathlib import Path


def export_numpy_to_binary(npy_path, output_dir="validation", num_frames=None):
    """
    Export a numpy array to binary format with metadata.

    Args:
        npy_path: Path to the .npy file
        output_dir: Output directory for .bin and .json files
        num_frames: Number of frames to export (None for all frames)
    """
    # Load the numpy array
    data = np.load(npy_path)

    # Slice the data if num_frames is specified
    if num_frames is not None:
        if len(data.shape) >= 2:
            # Assuming time dimension is the last dimension
            data = data[..., :num_frames]
        else:
            data = data[:num_frames]

    # Get file stem (filename without extension)
    file_stem = Path(npy_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Export binary data
    bin_path = output_path / f"{file_stem}.bin"
    data.astype(np.float16).tobytes()  # Convert to float16 for consistency
    with open(bin_path, "wb") as f:
        f.write(data.astype(np.float16).tobytes())

    # Export metadata
    metadata = {
        "shape": list(data.shape),
        "dtype": "float16",
        "byte_size": data.astype(np.float16).nbytes,
        "total_elements": int(data.size),
        "description": f"Binary export of {npy_path}",
    }

    json_path = output_path / f"{file_stem}_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {npy_path}:")
    print(f"  Binary: {bin_path}")
    print(f"  Metadata: {json_path}")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Size: {metadata['byte_size']} bytes")
    print()

    return bin_path, json_path


def main():
    """Main export function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Export numpy arrays to binary format for TypeScript consumption"
    )
    parser.add_argument(
        "-n",
        "--num-windows",
        type=int,
        default=10,
        help="Number of windows to export (default: 10)",
    )
    args = parser.parse_args()

    num_windows = args.num_windows
    frames_per_window_input = 200  # scaled_traces window size
    frames_per_window_output = 120  # model_outputs window size

    total_input_frames = num_windows * frames_per_window_input
    total_output_frames = num_windows * frames_per_window_output

    print(f"Exporting numpy arrays to binary format for TypeScript...")
    print(f"Number of windows: {num_windows}")
    print(f"Total input frames (scaled_traces): {total_input_frames}")
    print(f"Total output frames (model_outputs): {total_output_frames}")
    print()

    # Export both files
    validation_dir = Path("validation")

    # Check if files exist
    scaled_traces_path = validation_dir / "scaled_traces.npy"
    model_outputs_path = validation_dir / "model_outputs.npy"

    if not scaled_traces_path.exists():
        print(f"Error: {scaled_traces_path} not found!")
        return

    if not model_outputs_path.exists():
        print(f"Error: {model_outputs_path} not found!")
        return

    # Export the files with specified number of frames
    export_numpy_to_binary(scaled_traces_path, num_frames=total_input_frames)
    export_numpy_to_binary(model_outputs_path, num_frames=total_output_frames)

    # Create a summary metadata file
    summary = {
        "files": {
            "scaled_traces": {
                "binary": "scaled_traces.bin",
                "metadata": "scaled_traces_metadata.json",
                "description": "Input scaled traces data for the detection model",
            },
            "model_outputs": {
                "binary": "model_outputs.bin",
                "metadata": "model_outputs_metadata.json",
                "description": "Expected output from the detection model",
            },
        },
        "model_parameters": {
            "sample_size": 200,
            "num_output_locs": 120,  # sample_size - buffer_front_sample - buffer_end_sample
            "input_scale": 0.15887516,
            "inference_scaling_numerator": 12.6,
            "pre_median_frames": 1000,
            "buffer_front_sample": 40,
            "buffer_end_sample": 40,
            "num_windows_exported": num_windows,
            "total_input_frames_exported": total_input_frames,
            "total_output_frames_exported": total_output_frames,
        },
        "processing_notes": [
            "Use first 1000 frames to calculate inference scaling with inference_scaling.onnx",
            f"Process first {total_input_frames} frames ({num_windows} windows of 200) to get first {total_output_frames} outputs ({num_windows} windows of 120) with detect.onnx",
            "Input to detect.onnx should be: (traces - median) * input_scale * inference_scaling",
            "Input shape to detect.onnx: [num_channels, 1, 200]",
            "Output shape from detect.onnx: [num_channels, 1, 120]",
        ],
    }

    summary_path = validation_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to: {summary_path}")
    print("Export complete!")


if __name__ == "__main__":
    main()
