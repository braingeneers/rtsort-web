"""
Export the pytorch detection model to ONNX format and verify its outputs against Braindance and PyTorch implementations.
"""

from tqdm import tqdm
import torch
import numpy as np

from braindance.core.spikedetector.model import ModelSpikeSorter
from braindance.core.spikesorter.rt_sort import detect_sequences

import onnx
import onnxruntime

if __name__ == "__main__":
    print("Initializing Braindance Spike Detector ...")
    pytorch_model = ModelSpikeSorter.load("checkpoints/spikedetector/mea")
    pytorch_model.model.eval()  # Set model to evaluation mode

    """
    Expected output when debug = False:

    Saving traces:
    100%|██████████| 1/1 [00:04<00:00,  4.46s/it]
    Running detection model:
    Compiling detection model for 942 elecs ...
    Cannot compile detection model with torch_tensorrt because cannot load torch_tensorrt. Skipping NVIDIA compilation
    Allocating disk space to save model traces and outputs ...
    Inference scaling: 0.3761194029850746
    Running model ...
    100%|██████████| 832/832 [09:36<00:00,  1.44it/s]
    Detecting sequences
    100%|██████████| 942/942 [00:04<00:00, 211.68it/s]
    Detected 10 preliminary propagation sequences
    Extracting sequences' detections, intervals, and amplitudes
    
    100%|██████████| 10/10 [00:02<00:00,  4.03it/s]
    8 clusters remain after filtering
    Reassigning spikes to preliminary propagation sequences
    Initializing ...
    Sorting recording
    100%|██████████| 1000/1000 [00:00<00:00, 3377.24it/s]
    Extracting sequences' detections, intervals, and amplitudes
    
    100%|██████████| 7/7 [00:02<00:00,  2.80it/s]
    7 clusters remain after filtering
    Merging preliminary propagation sequences - first round
    
    100%|██████████| 7/7 [00:02<00:00,  3.14it/s]
    7 sequences after first merging
    Merging preliminary propagation sequences - second round ...
    
    RT-Sort detected 7 sequences
    """

    print("Detecting sequences in the first 5 minutes of a recording ...")
    rt_sort = detect_sequences(
        "data/MEA_rec_patch_ground_truth_cell7.raw.h5",
        "data/inter",
        pytorch_model,
        recording_window_ms=(0, 5 * 1000),
        device="cpu",
        verbose=True,
        debug=True,
        # num_processes=1,  # Uncomment for debugging
    )

    # Load outputs outputs from detect_sequences
    scaled_traces = np.load("data/inter/scaled_traces.npy")
    model_outputs = np.load("data/inter/model_outputs.npy")

    print("Exporting PyTorch model to ONNX format ...")
    torch.onnx.export(
        pytorch_model.model,
        torch.zeros(1, 1, pytorch_model.sample_size, dtype=torch.float16),
        str("public/models/detect-mea.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "sequence_length"},
            "output": {0: "batch_size", 2: "sequence_length"},
        },
        opset_version=12,
        verbose=False,
    )

    ort_session = onnxruntime.InferenceSession("public/models/detect-mea.onnx")

    print("Running PyTorch and ONNX models to verify outputs ...")

    # Convert input to torch tensor
    scaled_traces = torch.tensor(scaled_traces, dtype=torch.float16)

    # Get model parameters
    sample_size = pytorch_model.sample_size
    num_output_locs = pytorch_model.num_output_locs
    input_scale = pytorch_model.input_scale
    num_chans, rec_duration = scaled_traces.shape

    pre_median_frames = 1000
    inference_scaling_numerator = 12.6
    device = "cpu"

    # Calculate inference scaling based on initial window
    window = (
        scaled_traces[:, :pre_median_frames].to(torch.float32).cpu()
    )  # Cast to float32 and move to CPU
    if window.dtype != torch.float32:
        raise ValueError(
            f"Window tensor dtype is {window.dtype}, expected torch.float32"
        )
    iqrs = torch.quantile(window, 0.75, dim=1) - torch.quantile(window, 0.25, dim=1)
    median_iqr = torch.median(iqrs)
    inference_scaling = (
        inference_scaling_numerator / median_iqr if median_iqr != 0 else 1
    )

    # Define windows for processing
    all_start_frames = list(range(0, rec_duration - sample_size + 1, num_output_locs))[
        0:10
    ]
    output_duration = rec_duration - sample_size + 1
    outputs_all = np.zeros((num_chans, output_duration), dtype=np.float16)

    # Process each window
    with torch.no_grad():
        for start_frame in tqdm(all_start_frames):
            # Extract window
            traces_torch = scaled_traces[:, start_frame : start_frame + sample_size]

            # Subtract median for baseline correction
            traces_torch = (
                traces_torch - torch.median(traces_torch, dim=1, keepdim=True).values
            )

            # Run model on window and store output
            input_frame = traces_torch[:, None, :] * input_scale * inference_scaling

            pytorch_outputs = pytorch_model.model(input_frame)
            onnx_outputs = ort_session.run(
                ["output"],
                {"input": input_frame.numpy()},
            )[0]

            match = np.isclose(
                pytorch_outputs.detach().numpy(),
                model_outputs[
                    :, start_frame : start_frame + pytorch_model.num_output_locs
                ],
                rtol=1e-2,
            ).all()
            assert match, "PyTorch and Braindance outputs do not match!"

            match = np.isclose(
                pytorch_outputs.detach().numpy(),
                onnx_outputs,
                rtol=1e-2,
            ).all()
            assert match, "PyTorch and ONNX outputs do not match!"

    print("ONNX model exported and output validated against pytorch.")
