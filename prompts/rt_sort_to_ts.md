I've exported onnx models to calculate the inference_scaling factor from the first portion of a windo into models/inference_scaling.onnx and another model, models/detect.onnx that takes a window scaled by the inference_scaling coefficitn previously calculated and with some of the model parameters (see rt_sort.py for defaults) and yields outputs indicating where spikes are.

Generate a typescript file, detect.ts, that reads in scaled outputs from validation/scaled_outputs.npy, runs inference_scaling.onnx on the first 1000 window, and then proceeds to process just the first frame of 200 from validation/scaled_outputs.npy which should yield the first 120 in validation/model_outputs.npy.

If its challenging to process .npy files then generate a simple python script to export the .npy files (scaled_traces.npy and model_outputs.npy) as a flat binary file (.bin) or possibly better something that onnx torch can read in typescript.
