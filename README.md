# rtsort-web

Single page web application that detects and plots spikes in ephys data streamed from a .h5 file.

This is achieved by running [RTSort](https://braingeneers.github.io/braindance/docs/RT-sort/introduction) in the browser using [h5wasm](https://github.com/usnistgov/h5wasm) to read local ephys data (.h5) files and [ONNX](https://onnxruntime.ai/) to run the spike detetion model.

# Install

```
python3.11 -m venv venv
pip install -e ./braindance[rt-sort]

```

# References

[MEA Recordings](https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset)
