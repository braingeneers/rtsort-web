#!/usr/bin/env python3

import typer
from pathlib import Path
import json
import torch
import onnx
import onnxruntime
import numpy as np
import sys  # For sys.stderr

# Attempt to import ModelSpikeSorter from braindance library
try:
    from braindance.core.spikedetector.model import ModelSpikeSorter
except ImportError:
    # This will be handled within the command, allowing the script to be imported
    # or run with --help even if braindance is not perfectly set up.
    ModelSpikeSorter = None
app = typer.Typer(
    help="CLI tool to convert PyTorch ModelSpikeSorter models to ONNX and validate them."
)


def get_device() -> torch.device:
    """Returns the appropriate torch device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.command()
def convert(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the init_dict.json and state_dict.pt files for the model.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        show_default=False,
        resolve_path=True,
    ),
    output_onnx_path: Path = typer.Argument(
        ...,
        help="Path to save the exported ONNX model (e.g., model.onnx).",
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        show_default=False,
    ),
    dummy_sequence_length: int = typer.Option(
        2048, help="Sequence length for the dummy input used in ONNX export."
    ),
    opset_version: int = typer.Option(11, help="ONNX opset version to use for export."),
):
    """
    Converts a PyTorch ModelSpikeSorter model (state_dict.pt and init_dict.json) to an ONNX file.
    """
    if ModelSpikeSorter is None:
        typer.secho(
            "Error: braindance.core.spikedetector.ModelSpikeSorter could not be imported. "
            "Please ensure the 'braindance' library is installed correctly and accessible in your PYTHONPATH.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Loading model from: {model_path}")
    try:
        with open(model_path / "init_dict.json", "r") as f:
            init_dict = json.load(f)
        model = ModelSpikeSorter(**init_dict)
        state_dict = torch.load(model_path / "state_dict.pt", map_location="cpu")
        model.load_state_dict(state_dict)
    except TypeError as e:
        typer.secho(
            f"Error initializing ModelSpikeSorter model: {e}. Check if init_args match ModelSpikeSorter constructor.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"Error initializing ModelSpikeSorter model: {e}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    device = get_device()
    typer.echo(f"Using device: {device}")
    try:
        # # Move the model to GPU if needed
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        model.eval()  # Set model to evaluation mode
        # Convert all parameters to float32
        model = model.float()  # This casts all parameters to torch.float32
    except Exception as e:
        typer.secho(f"Error loading model: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("PyTorch model loaded successfully and set to evaluation mode.")

    n_channels = init_dict.get("num_channels_in")
    if not isinstance(n_channels, int) or n_channels <= 0:
        typer.secho(
            f"Error: 'n_channels' in init_args must be a positive integer, got {n_channels}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    batch_size = 1
    dummy_input = torch.randn(
        batch_size, n_channels, dummy_sequence_length, device=device
    )
    typer.echo(
        f"Preparing for ONNX export with dummy input of shape: {list(dummy_input.shape)}"
    )

    try:
        output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        typer.secho(
            f"Error creating output directory {output_onnx_path.parent}: {e}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Exporting model to ONNX format at: {output_onnx_path}")
    try:
        torch.onnx.export(
            model.model,
            dummy_input,
            str(output_onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "sequence_length"},
                "output": {0: "batch_size", 2: "sequence_length"},
            },
            opset_version=opset_version,
            verbose=False,
        )
        typer.secho(
            f"Model successfully exported to {output_onnx_path}", fg=typer.colors.GREEN
        )
    except Exception as e:
        typer.secho(f"Error during ONNX export: {e}", fg=typer.colors.RED)
        if output_onnx_path.exists():
            try:
                output_onnx_path.unlink()
                typer.echo(f"Cleaned up partially created file: {output_onnx_path}")
            except Exception as unlink_e:
                typer.secho(
                    f"Warning: Could not clean up {output_onnx_path}: {unlink_e}",
                    fg=typer.colors.YELLOW,
                )
        raise typer.Exit(code=1)

    typer.echo("Verifying the ONNX model...")
    try:
        onnx_model = onnx.load(str(output_onnx_path))
        onnx.checker.check_model(onnx_model)
        typer.echo("ONNX model structure is valid.")
    except Exception as e:
        typer.secho(
            f"Warning: ONNX model verification failed: {e}", fg=typer.colors.YELLOW
        )
        typer.secho(
            "The ONNX file was created, but it might have issues.",
            fg=typer.colors.YELLOW,
        )


def _evaluate_stub(output_data: np.ndarray, onnx_path: Path, nwb_file_path: Path):
    """Stub function to 'evaluate' the model output."""
    typer.echo("\n--- Evaluation Stub ---")
    typer.echo(f"Validation based on ONNX model: {onnx_path}")
    typer.echo(f"Input NWB file (data was stubbed): {nwb_file_path}")
    typer.echo(f"Model output data shape: {output_data.shape}")
    typer.echo(f"Model output data type: {output_data.dtype}")

    if output_data.size > 0:
        typer.echo(
            f"Output data min: {output_data.min():.4f}, max: {output_data.max():.4f}, mean: {output_data.mean():.4f}"
        )
    else:
        typer.echo("Output data is empty.")

    typer.echo(
        "Evaluation logic (stubbed): Processing complete. No actual metrics calculated."
    )
    typer.echo("--- End Evaluation Stub ---\n")


@app.command()
def validate(
    onnx_path: Path = typer.Option(
        ...,
        help="Path to the ONNX model file to validate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        show_default=False,
    ),
    nwb_file_path: Path = typer.Option(
        ...,
        help="Path to a sample NWB file. (Note: NWB data processing is currently STUBBED).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        show_default=False,
    ),
):
    """
    Validates an ONNX spike detection model using a sample NWB file (data processing stubbed).
    """
    typer.echo(f"Loading ONNX model from: {onnx_path}")
    try:
        ort_session = onnxruntime.InferenceSession(str(onnx_path))
        typer.echo("ONNX model loaded successfully using onnxruntime.")
    except Exception as e:
        typer.secho(
            f"Error loading ONNX model with onnxruntime: {e}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    if not ort_session.get_inputs():
        typer.secho("Error: ONNX model has no inputs defined.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    input_meta = ort_session.get_inputs()[0]
    input_name = input_meta.name
    onnx_input_shape = input_meta.shape
    typer.echo(
        f"Model expected input name: '{input_name}', symbolic shape: {onnx_input_shape}"
    )

    if (
        len(onnx_input_shape) != 3
        or not isinstance(onnx_input_shape[1], int)
        or onnx_input_shape[1] <= 0
    ):
        typer.secho(
            f"Error: Could not determine a fixed, positive number of channels from ONNX model input shape: {onnx_input_shape}. "
            "Expected format like [None, num_channels, None] or [1, num_channels, seq_length].",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    expected_n_channels_from_onnx = onnx_input_shape[1]
    typer.echo(f"ONNX model expects {expected_n_channels_from_onnx} input channels.")

    model_sequence_length_info = onnx_input_shape[2]
    if (
        isinstance(model_sequence_length_info, str)
        or model_sequence_length_info is None
    ):
        sample_sequence_length = 2048
        typer.echo(
            f"ONNX model has dynamic sequence length (symbol: '{model_sequence_length_info}'). "
            f"Using sample sequence length for stub: {sample_sequence_length}"
        )
    elif isinstance(model_sequence_length_info, int) and model_sequence_length_info > 0:
        sample_sequence_length = model_sequence_length_info
        typer.echo(f"ONNX model has fixed sequence length: {sample_sequence_length}")
    else:
        typer.secho(
            f"Error: Unexpected type or value for sequence length in ONNX model shape: {model_sequence_length_info}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.secho(
        f"\nNWB data loading from '{nwb_file_path}' is STUBBED.", fg=typer.colors.YELLOW
    )
    typer.secho(
        "Using randomly generated data matching model's expected input dimensions for this validation.",
        fg=typer.colors.YELLOW,
    )

    batch_size = 1
    stub_input_data_shape = (
        batch_size,
        expected_n_channels_from_onnx,
        sample_sequence_length,
    )
    typer.echo(f"Creating STUB input ephys data of shape {stub_input_data_shape}.")
    ephys_data_segment = np.random.randn(*stub_input_data_shape).astype(np.float32)

    typer.echo(
        f"Running inference with STUBBED data of shape: {ephys_data_segment.shape}"
    )
    try:
        ort_inputs = {input_name: ephys_data_segment}
        ort_outs = ort_session.run(None, ort_inputs)
        if not ort_outs:
            typer.secho(
                "Error: ONNX model inference returned no outputs.", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        output_data = ort_outs[0]
    except Exception as e:
        typer.secho(f"Error during ONNX inference: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("ONNX model inference complete.")
    _evaluate_stub(output_data, onnx_path=onnx_path, nwb_file_path=nwb_file_path)

    typer.secho(
        f"Validation process for {onnx_path} (using stubbed NWB data) finished.",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    if ModelSpikeSorter is None:
        print(
            "Warning: The 'braindance' library (specifically ModelSpikeSorter) could not be imported. "
            "The 'convert' command will fail. Ensure 'braindance' is installed and in PYTHONPATH.",
            file=sys.stderr,
        )
    app()
