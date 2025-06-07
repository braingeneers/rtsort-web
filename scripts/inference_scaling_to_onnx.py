import torch
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np


def create_inference_scaling_onnx_model():
    """
    Creates an ONNX model that computes inference scaling from a window of data.

    The model takes:
    - window: input tensor of shape (num_channels, frames)
    - inference_scaling_numerator: scalar value

    Returns:
    - inference_scaling: scalar output
    """

    # Define inputs
    window_input = helper.make_tensor_value_info(
        "window", TensorProto.FLOAT16, ["num_channels", "frames"]
    )
    numerator_input = helper.make_tensor_value_info(
        "inference_scaling_numerator", TensorProto.FLOAT, []
    )

    # Define output
    scaling_output = helper.make_tensor_value_info(
        "inference_scaling", TensorProto.FLOAT, []
    )

    # Create constant tensors
    zero_const = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    one_const = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    two_const = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    q25_const = helper.make_tensor("q25_val", TensorProto.FLOAT, [], [0.25])
    q75_const = helper.make_tensor("q75_val", TensorProto.FLOAT, [], [0.75])
    axis_1_const = helper.make_tensor("axis_1", TensorProto.INT64, [], [1])
    zero_int_const = helper.make_tensor("zero_int", TensorProto.INT64, [], [0])
    scalar_shape_const = helper.make_tensor("scalar_shape", TensorProto.INT64, [0], [])

    # Define the computation nodes
    nodes = [
        # Cast float16 window input to float32 for computation
        helper.make_node(
            "Cast",
            inputs=["window"],
            outputs=["window_f32"],
            to=TensorProto.FLOAT,
        ),
        # Get the shape of window along dimension 1 (number of frames)
        helper.make_node("Shape", inputs=["window_f32"], outputs=["window_shape"]),
        helper.make_node(
            "Gather",
            inputs=["window_shape", "axis_1"],
            outputs=["window_frames_count_scalar"],
        ),
        # Convert scalar to 1D tensor for TopK
        helper.make_node(
            "Unsqueeze",
            inputs=["window_frames_count_scalar"],
            outputs=["window_frames_count"],
            axes=[0],
        ),
        # Sort along axis 1 (frames dimension) for each channel
        helper.make_node(
            "TopK",
            inputs=["window_f32", "window_frames_count"],
            outputs=["sorted_values", "sorted_indices"],
            axis=1,
            largest=0,  # Sort in ascending order
            sorted=1,
        ),
        # Convert shape to float for calculations
        helper.make_node(
            "Cast",
            inputs=["window_frames_count_scalar"],
            outputs=["num_frames_float"],
            to=TensorProto.FLOAT,
        ),
        # Calculate quantile indices: q * (n-1)
        helper.make_node(
            "Sub", inputs=["num_frames_float", "one"], outputs=["n_minus_1"]
        ),
        helper.make_node(
            "Mul", inputs=["q25_val", "n_minus_1"], outputs=["q25_index_float"]
        ),
        helper.make_node(
            "Mul", inputs=["q75_val", "n_minus_1"], outputs=["q75_index_float"]
        ),
        # For linear interpolation, we need floor and ceil indices
        # Q25 interpolation
        helper.make_node(
            "Floor", inputs=["q25_index_float"], outputs=["q25_floor_float"]
        ),
        helper.make_node(
            "Cast",
            inputs=["q25_floor_float"],
            outputs=["q25_floor"],
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Add", inputs=["q25_floor", "axis_1"], outputs=["q25_ceil"]  # floor + 1
        ),
        # Clamp ceil to valid range
        helper.make_node(
            "Sub",
            inputs=["window_frames_count_scalar", "axis_1"],
            outputs=["max_index"],
        ),
        helper.make_node(
            "Min", inputs=["q25_ceil", "max_index"], outputs=["q25_ceil_clamped"]
        ),
        # Q75 interpolation
        helper.make_node(
            "Floor", inputs=["q75_index_float"], outputs=["q75_floor_float"]
        ),
        helper.make_node(
            "Cast",
            inputs=["q75_floor_float"],
            outputs=["q75_floor"],
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Add", inputs=["q75_floor", "axis_1"], outputs=["q75_ceil"]  # floor + 1
        ),
        helper.make_node(
            "Min", inputs=["q75_ceil", "max_index"], outputs=["q75_ceil_clamped"]
        ),
        # Calculate interpolation fractions
        helper.make_node(
            "Sub", inputs=["q25_index_float", "q25_floor_float"], outputs=["q25_frac"]
        ),
        helper.make_node(
            "Sub", inputs=["q75_index_float", "q75_floor_float"], outputs=["q75_frac"]
        ),
        helper.make_node(
            "Sub", inputs=["one", "q25_frac"], outputs=["q25_one_minus_frac"]
        ),
        helper.make_node(
            "Sub", inputs=["one", "q75_frac"], outputs=["q75_one_minus_frac"]
        ),
        # Get values at floor and ceil indices
        helper.make_node(
            "Gather",
            inputs=["sorted_values", "q25_floor"],
            outputs=["q25_floor_values"],
            axis=1,
        ),
        helper.make_node(
            "Gather",
            inputs=["sorted_values", "q25_ceil_clamped"],
            outputs=["q25_ceil_values"],
            axis=1,
        ),
        helper.make_node(
            "Gather",
            inputs=["sorted_values", "q75_floor"],
            outputs=["q75_floor_values"],
            axis=1,
        ),
        helper.make_node(
            "Gather",
            inputs=["sorted_values", "q75_ceil_clamped"],
            outputs=["q75_ceil_values"],
            axis=1,
        ),
        # Linear interpolation: floor_val * (1-frac) + ceil_val * frac
        helper.make_node(
            "Mul",
            inputs=["q25_floor_values", "q25_one_minus_frac"],
            outputs=["q25_floor_contrib"],
        ),
        helper.make_node(
            "Mul", inputs=["q25_ceil_values", "q25_frac"], outputs=["q25_ceil_contrib"]
        ),
        helper.make_node(
            "Add",
            inputs=["q25_floor_contrib", "q25_ceil_contrib"],
            outputs=["q25_values"],
        ),
        helper.make_node(
            "Mul",
            inputs=["q75_floor_values", "q75_one_minus_frac"],
            outputs=["q75_floor_contrib"],
        ),
        helper.make_node(
            "Mul", inputs=["q75_ceil_values", "q75_frac"], outputs=["q75_ceil_contrib"]
        ),
        helper.make_node(
            "Add",
            inputs=["q75_floor_contrib", "q75_ceil_contrib"],
            outputs=["q75_values"],
        ),
        # Compute IQR = q75 - q25
        helper.make_node("Sub", inputs=["q75_values", "q25_values"], outputs=["iqrs"]),
        # Sort IQRs for median calculation
        helper.make_node("Shape", inputs=["iqrs"], outputs=["iqrs_shape"]),
        helper.make_node(
            "Gather", inputs=["iqrs_shape", "zero_int"], outputs=["iqrs_length_scalar"]
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=["iqrs_length_scalar"],
            outputs=["iqrs_k_tensor"],
            axes=[0],
        ),
        helper.make_node(
            "TopK",
            inputs=["iqrs", "iqrs_k_tensor"],
            outputs=["iqrs_sorted", "iqrs_sort_indices"],
            axis=0,
            largest=0,
            sorted=1,
        ),
        # Calculate median indices for the sorted IQRs
        helper.make_node(
            "Cast",
            inputs=["iqrs_length_scalar"],
            outputs=["iqrs_length_float"],
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Sub", inputs=["iqrs_length_float", "one"], outputs=["n_minus_1_median"]
        ),
        helper.make_node(
            "Div", inputs=["n_minus_1_median", "two"], outputs=["lower_idx_float"]
        ),
        helper.make_node(
            "Floor", inputs=["lower_idx_float"], outputs=["lower_idx_floor"]
        ),
        helper.make_node(
            "Cast",
            inputs=["lower_idx_floor"],
            outputs=["lower_median_idx"],
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Div", inputs=["iqrs_length_float", "two"], outputs=["upper_idx_float"]
        ),
        helper.make_node(
            "Floor", inputs=["upper_idx_float"], outputs=["upper_idx_floor"]
        ),
        helper.make_node(
            "Cast",
            inputs=["upper_idx_floor"],
            outputs=["upper_median_idx"],
            to=TensorProto.INT64,
        ),
        # Get values at median indices
        helper.make_node(
            "Gather",
            inputs=["iqrs_sorted", "lower_median_idx"],
            outputs=["lower_median_value"],
        ),
        helper.make_node(
            "Gather",
            inputs=["iqrs_sorted", "upper_median_idx"],
            outputs=["upper_median_value"],
        ),
        # Average for median
        helper.make_node(
            "Add",
            inputs=["lower_median_value", "upper_median_value"],
            outputs=["sum_median_values"],
        ),
        helper.make_node(
            "Div", inputs=["sum_median_values", "two"], outputs=["median_iqr"]
        ),
        # Check if median_iqr != 0
        helper.make_node("Equal", inputs=["median_iqr", "zero"], outputs=["is_zero"]),
        # Compute numerator / median_iqr
        helper.make_node(
            "Div",
            inputs=["inference_scaling_numerator", "median_iqr"],
            outputs=["division_result"],
        ),
        # Select between division_result and 1 based on whether median_iqr is zero
        helper.make_node(
            "Where",
            inputs=["is_zero", "one", "division_result"],
            outputs=["inference_scaling_temp"],
        ),
        # Ensure output is a scalar with well-defined shape
        helper.make_node(
            "Reshape",
            inputs=["inference_scaling_temp", "scalar_shape"],
            outputs=["inference_scaling"],
        ),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        "inference_scaling_computation",
        [window_input, numerator_input],
        [scaling_output],
        [
            zero_const,
            one_const,
            two_const,
            q25_const,
            q75_const,
            axis_1_const,
            zero_int_const,
            scalar_shape_const,
        ],
    )

    # Create the model with explicit IR version
    model = helper.make_model(graph, producer_name="inference_scaling")
    model.opset_import[0].version = 12
    model.ir_version = 8  # Set IR version to 8 for compatibility

    return model


def save_inference_scaling_model(filename="public/models/inference_scaling.onnx"):
    """Save the inference scaling ONNX model to file."""
    model = create_inference_scaling_onnx_model()
    onnx.save(model, filename)
    print(f"Inference scaling ONNX model saved to {filename}")
    return model


def test_inference_scaling_onnx():
    """Test the ONNX model against the original PyTorch computation."""
    import onnxruntime as ort
    import torch

    # Create and save the model
    model = save_inference_scaling_model()

    # Create test data
    torch.manual_seed(42)
    window = torch.randn(10, 1000)  # 10 channels, 1000 frames
    inference_scaling_numerator = 12.6

    # Original PyTorch computation
    iqrs = torch.quantile(window, 0.75, dim=1) - torch.quantile(window, 0.25, dim=1)
    median_iqr = torch.median(iqrs)
    torch_result = inference_scaling_numerator / median_iqr if median_iqr != 0 else 1

    # ONNX computation
    session = ort.InferenceSession("models/inference_scaling.onnx")
    onnx_result = session.run(
        ["inference_scaling"],
        {
            "window": window.numpy().astype(np.float16),
            "inference_scaling_numerator": np.array(
                inference_scaling_numerator, dtype=np.float32
            ),
        },
    )[0]

    print(f"PyTorch result: {torch_result.item()}")
    print(f"ONNX result: {onnx_result.item()}")
    print(
        f"Results match (rtol=1e-6): {np.isclose(torch_result.item(), onnx_result.item(), rtol=1e-6)}"
    )
    print(
        f"Results match (rtol=1e-2): {np.isclose(torch_result.item(), onnx_result.item(), rtol=1e-2)}"
    )
    print(
        f"Relative difference: {abs(torch_result.item() - onnx_result.item()) / torch_result.item():.6f}"
    )

    return abs(torch_result.item() - onnx_result.item()) / torch_result.item() < 0.05


def test_inference_scaling_onnx_on_scaled_traces():
    scaled_traces = np.load("validation/scaled_traces.npy")

    # Convert to torch to match computation in ONNX model
    scaled_traces = torch.tensor(scaled_traces, dtype=torch.float16)

    # Specify the model and parameters
    inference_scaling_numerator = 12.6
    pre_median_frames = 1000

    # Calculate inference scaling based on initial window
    window = scaled_traces[:, :pre_median_frames]
    iqrs = torch.quantile(window.to(torch.float), 0.75, dim=1) - torch.quantile(
        window.to(torch.float), 0.25, dim=1
    )
    median_iqr = torch.median(iqrs)
    torch_result = inference_scaling_numerator / median_iqr if median_iqr != 0 else 1
    print(f"PyTorch result: {torch_result}")

    # ONNX computation
    session = ort.InferenceSession("public/models/inference_scaling.onnx")
    onnx_result = session.run(
        ["inference_scaling"],
        {
            "window": window.numpy(),
            "inference_scaling_numerator": np.array(
                inference_scaling_numerator, dtype=np.float32
            ),
        },
    )[0]
    print(f"ONNX result: {onnx_result.item()}")
    print(
        f"Results match (rtol=1e-6): {np.isclose(torch_result.item(), onnx_result.item(), rtol=1e-6)}"
    )
    print(
        f"Results match (rtol=1e-2): {np.isclose(torch_result.item(), onnx_result.item(), rtol=1e-2)}"
    )
    print(
        f"Relative difference: {abs(torch_result.item() - onnx_result.item()) / torch_result.item():.6f}"
    )


if __name__ == "__main__":
    print("Testing synthetic data:")
    test_inference_scaling_onnx()
    print("\nTesting real scaled_traces data:")
    test_inference_scaling_onnx_on_scaled_traces()
