# rtsort-web

Single page web application that runs [RTSort](https://braingeneers.github.io/braindance/docs/RT-sort/introduction) in the browser using [h5wasm](https://github.com/usnistgov/h5wasm) to stream local ephys data (.h5) files and [ONNX](https://onnxruntime.ai) to run the spike detetion model.

[Demo](https://braingeneers.github.io/rtsort-web)

![Alt text](screenshot.png?raw=true 'RT-Sort Web')

## Status

Proof of concept that runs RT-Sort pre-processing and spike detection for raw [Maxell Biosystems](https://www.mxwbio.com) recordings. Pre-processing is currently in javascript and could easily be moved into ONNX improving performance. The model output concordant with RT-Sort python to 3 decimal places using the CPU and 1 decimal place using the M3 GPU (see the browser console for concordance details). Spike sorting is TBD. Tested in Chrome and Safari on a MacBook M3.

## Install

To run the [Vue](https://vuejs.org) based web application:

```
npm install
npm run dev
```

To use the scripts to export a model to ONNX and subset files:

```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
