"use strict";
/**
 * TypeScript implementation for ONNX-based spike detection inference
 *
 * This script:
 * 1. Loads scaled traces from binary file
 * 2. Runs inference_scaling.onnx on first 1000 frames to calculate scaling factor
 * 3. Processes first 200 frames with detect.onnx to yield first 120 outputs
 * 4. Compares results with expected model outputs
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var ort = require("onnxruntime-node");
var fs = require("fs");
var util_1 = require("util");
var readFile = (0, util_1.promisify)(fs.readFile);
var Float16Array = /** @class */ (function () {
    function Float16Array(data) {
        if (typeof data === "number") {
            this.length = data;
            this.buffer = new ArrayBuffer(data * 2); // 2 bytes per float16
        }
        else {
            this.buffer = data;
            this.length = data.byteLength / 2;
        }
        this.view = new DataView(this.buffer);
    }
    // Convert float16 to float32 (IEEE 754 implementation)
    Float16Array.prototype.float16ToFloat32 = function (value) {
        var sign = (value >> 15) & 0x1;
        var exponent = (value >> 10) & 0x1f;
        var mantissa = value & 0x3ff;
        if (exponent === 0) {
            if (mantissa === 0) {
                return sign ? -0 : 0;
            }
            else {
                // Subnormal number
                return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
            }
        }
        else if (exponent === 31) {
            return mantissa ? NaN : sign ? -Infinity : Infinity;
        }
        else {
            return ((sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024));
        }
    };
    // Convert float32 to float16 (IEEE 754 implementation)
    Float16Array.float32ToFloat16 = function (value) {
        if (isNaN(value))
            return 0x7e00; // NaN
        if (!isFinite(value))
            return value > 0 ? 0x7c00 : 0xfc00; // ±Infinity
        var sign = value < 0 ? 1 : 0;
        var absValue = Math.abs(value);
        if (absValue === 0)
            return sign << 15; // ±0
        // Handle very small numbers (subnormal in float16)
        if (absValue < Math.pow(2, -14)) {
            var mantissa = Math.round(absValue / Math.pow(2, -24));
            return (sign << 15) | mantissa;
        }
        // Handle very large numbers (clamp to max float16)
        if (absValue >= 65504) {
            return (sign << 15) | 0x7bff; // Max finite float16
        }
        // Normal case
        var exponent = Math.floor(Math.log2(absValue));
        var normalizedMantissa = absValue / Math.pow(2, exponent) - 1;
        var mantissaBits = Math.round(normalizedMantissa * 1024);
        var exponentBits = exponent + 15;
        return (sign << 15) | (exponentBits << 10) | mantissaBits;
    };
    Float16Array.prototype.get = function (index) {
        if (index < 0 || index >= this.length) {
            throw new Error("Index ".concat(index, " out of bounds for Float16Array of length ").concat(this.length));
        }
        var uint16Value = this.view.getUint16(index * 2, true); // little endian
        return this.float16ToFloat32(uint16Value);
    };
    Float16Array.prototype.toFloat32Array = function () {
        var result = new Float32Array(this.length);
        for (var i = 0; i < this.length; i++) {
            result[i] = this.get(i);
        }
        return result;
    };
    // Get a slice as Float32Array for efficiency
    Float16Array.prototype.slice = function (start, end) {
        var endIndex = end !== null && end !== void 0 ? end : this.length;
        var sliceLength = endIndex - start;
        var result = new Float32Array(sliceLength);
        for (var i = 0; i < sliceLength; i++) {
            result[i] = this.get(start + i);
        }
        return result;
    };
    return Float16Array;
}());
function loadBinaryData(filePath, metadata) {
    return __awaiter(this, void 0, void 0, function () {
        var buffer;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    console.log("Loading binary data from ".concat(filePath, "..."));
                    return [4 /*yield*/, readFile(filePath)];
                case 1:
                    buffer = _a.sent();
                    // Verify file size matches metadata
                    if (buffer.byteLength !== metadata.byte_size) {
                        throw new Error("File size mismatch: expected ".concat(metadata.byte_size, ", got ").concat(buffer.byteLength));
                    }
                    return [2 /*return*/, new Float16Array(buffer.buffer)];
            }
        });
    });
}
function loadMetadata(filePath) {
    return __awaiter(this, void 0, void 0, function () {
        var content;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, readFile(filePath, "utf-8")];
                case 1:
                    content = _a.sent();
                    return [2 /*return*/, JSON.parse(content)];
            }
        });
    });
}
function calculateMedian(arr) {
    var sorted = Array.from(arr).sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    else {
        return sorted[mid];
    }
}
function calculateQuantile(arr, q) {
    var sorted = Array.from(arr).sort(function (a, b) { return a - b; });
    var index = q * (sorted.length - 1);
    var lower = Math.floor(index);
    var upper = Math.ceil(index);
    if (lower === upper) {
        return sorted[lower];
    }
    else {
        var weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
}
function reshapeData(data, shape) {
    if (shape.length !== 2) {
        throw new Error("Only 2D reshaping supported");
    }
    var numChannels = shape[0], timePoints = shape[1];
    var channels = [];
    for (var ch = 0; ch < numChannels; ch++) {
        var channelData = new Float32Array(timePoints);
        for (var t = 0; t < timePoints; t++) {
            channelData[t] = data.get(ch * timePoints + t);
        }
        // Convert back to Float16Array for consistency
        var channelBuffer = new ArrayBuffer(timePoints * 2);
        var channelView = new DataView(channelBuffer);
        for (var t = 0; t < timePoints; t++) {
            // Simplified float32 to float16 conversion (may lose precision)
            var value = channelData[t];
            var uint16Value = Math.round(value * 100) / 100; // Simplified
            channelView.setUint16(t * 2, uint16Value, true);
        }
        channels.push(new Float16Array(channelBuffer));
    }
    return channels;
}
function runInferenceScaling(scaledTraces, shape, params) {
    return __awaiter(this, void 0, void 0, function () {
        var numChannels, timePoints, pre_median_frames, inference_scaling_numerator, window, ch, t, session, windowUint16, i, windowTensor, numeratorTensor, feeds, results, inferenceScaling, iqrs, ch, channelWindow, q25, q75, medianIqr, manualInferenceScaling, error_1, iqrs, ch, channelWindow, q25, q75, medianIqr, inferenceScaling;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    console.log("Running inference scaling calculation...");
                    numChannels = shape[0], timePoints = shape[1];
                    pre_median_frames = params.pre_median_frames, inference_scaling_numerator = params.inference_scaling_numerator;
                    window = new Float32Array(numChannels * pre_median_frames);
                    for (ch = 0; ch < numChannels; ch++) {
                        for (t = 0; t < pre_median_frames; t++) {
                            window[ch * pre_median_frames + t] = scaledTraces.get(ch * timePoints + t);
                        }
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, , 5]);
                    return [4 /*yield*/, ort.InferenceSession.create("models/inference_scaling.onnx")];
                case 2:
                    session = _a.sent();
                    windowUint16 = new Uint16Array(numChannels * pre_median_frames);
                    for (i = 0; i < window.length; i++) {
                        windowUint16[i] = Float16Array.float32ToFloat16(window[i]);
                    }
                    windowTensor = new ort.Tensor("float16", windowUint16, [
                        numChannels,
                        pre_median_frames,
                    ]);
                    numeratorTensor = new ort.Tensor("float32", new Float32Array([inference_scaling_numerator]), []);
                    feeds = {
                        window: windowTensor,
                        inference_scaling_numerator: numeratorTensor,
                    };
                    return [4 /*yield*/, session.run(feeds)];
                case 3:
                    results = _a.sent();
                    inferenceScaling = results["inference_scaling"].data[0];
                    console.log("ONNX inference scaling factor: ".concat(inferenceScaling));
                    iqrs = new Float32Array(numChannels);
                    for (ch = 0; ch < numChannels; ch++) {
                        channelWindow = window.slice(ch * pre_median_frames, (ch + 1) * pre_median_frames);
                        q25 = calculateQuantile(channelWindow, 0.25);
                        q75 = calculateQuantile(channelWindow, 0.75);
                        iqrs[ch] = q75 - q25;
                    }
                    medianIqr = calculateMedian(iqrs);
                    manualInferenceScaling = medianIqr !== 0 ? inference_scaling_numerator / medianIqr : 1;
                    console.log("Manual inference scaling factor: ".concat(manualInferenceScaling));
                    console.log("Difference: ".concat(Math.abs(inferenceScaling - manualInferenceScaling)));
                    return [2 /*return*/, inferenceScaling];
                case 4:
                    error_1 = _a.sent();
                    console.warn("ONNX inference scaling failed, calculating manually:", error_1);
                    iqrs = new Float32Array(numChannels);
                    for (ch = 0; ch < numChannels; ch++) {
                        channelWindow = window.slice(ch * pre_median_frames, (ch + 1) * pre_median_frames);
                        q25 = calculateQuantile(channelWindow, 0.25);
                        q75 = calculateQuantile(channelWindow, 0.75);
                        iqrs[ch] = q75 - q25;
                    }
                    medianIqr = calculateMedian(iqrs);
                    inferenceScaling = medianIqr !== 0 ? inference_scaling_numerator / medianIqr : 1;
                    console.log("Manual inference scaling factor: ".concat(inferenceScaling));
                    return [2 /*return*/, inferenceScaling];
                case 5: return [2 /*return*/];
            }
        });
    });
}
function runDetection(scaledTraces, shape, params, inferenceScaling) {
    return __awaiter(this, void 0, void 0, function () {
        var numChannels, timePoints, sample_size, num_output_locs, input_scale, windowData, ch, t, processedWindow, ch, channelData, median, t, baselineSubtracted, session, inputTensor, feeds, results, outputs, error_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    console.log("Running spike detection...");
                    numChannels = shape[0], timePoints = shape[1];
                    sample_size = params.sample_size, num_output_locs = params.num_output_locs, input_scale = params.input_scale;
                    windowData = new Float32Array(numChannels * sample_size);
                    for (ch = 0; ch < numChannels; ch++) {
                        for (t = 0; t < sample_size; t++) {
                            windowData[ch * sample_size + t] = scaledTraces.get(ch * timePoints + t);
                        }
                    }
                    processedWindow = new Float32Array(numChannels * sample_size);
                    for (ch = 0; ch < numChannels; ch++) {
                        channelData = windowData.slice(ch * sample_size, (ch + 1) * sample_size);
                        median = calculateMedian(channelData);
                        for (t = 0; t < sample_size; t++) {
                            baselineSubtracted = channelData[t] - median;
                            processedWindow[ch * sample_size + t] =
                                baselineSubtracted * input_scale * inferenceScaling;
                        }
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, , 5]);
                    return [4 /*yield*/, ort.InferenceSession.create("models/detect.onnx")];
                case 2:
                    session = _a.sent();
                    inputTensor = new ort.Tensor("float32", processedWindow, [
                        numChannels,
                        1,
                        sample_size,
                    ]);
                    feeds = { input: inputTensor };
                    return [4 /*yield*/, session.run(feeds)];
                case 3:
                    results = _a.sent();
                    outputs = results["output"].data;
                    console.log("Detection completed. Output shape: [".concat(numChannels, ", 1, ").concat(num_output_locs, "]"));
                    console.log("First few outputs: [".concat(Array.from(outputs.slice(0, 10))
                        .map(function (x) { return x.toFixed(4); })
                        .join(", "), "...]"));
                    return [2 /*return*/, outputs];
                case 4:
                    error_2 = _a.sent();
                    console.error("ONNX detection failed:", error_2);
                    throw error_2;
                case 5: return [2 /*return*/];
            }
        });
    });
}
function compareOutputs(predicted, expected, shape) {
    console.log("\nComparing predicted vs expected outputs...");
    var _a = [shape[0], 120], numChannels = _a[0], numOutputLocs = _a[1]; // First 120 outputs
    var tolerance = 1e-2;
    var totalElements = 0;
    var matchingElements = 0;
    var maxDifference = 0;
    var totalAbsoluteDifference = 0;
    for (var ch = 0; ch < numChannels; ch++) {
        for (var t = 0; t < numOutputLocs; t++) {
            var predIdx = ch * numOutputLocs + t;
            var expIdx = ch * shape[1] + t; // expected has full time dimension
            if (predIdx < predicted.length && expIdx < expected.length) {
                var predValue = predicted[predIdx];
                var expValue = expected.get(expIdx);
                var difference = Math.abs(predValue - expValue);
                totalElements++;
                totalAbsoluteDifference += difference;
                if (difference <= tolerance) {
                    matchingElements++;
                }
                if (difference > maxDifference) {
                    maxDifference = difference;
                }
                // Log first few comparisons for debugging
                if (totalElements <= 5) {
                    console.log("  Element ".concat(totalElements, ": predicted=").concat(predValue.toFixed(4), ", expected=").concat(expValue.toFixed(4), ", diff=").concat(difference.toFixed(4)));
                }
            }
        }
    }
    var matchPercentage = (matchingElements / totalElements) * 100;
    var avgAbsDifference = totalAbsoluteDifference / totalElements;
    console.log("\nComparison Results:");
    console.log("  Total elements compared: ".concat(totalElements));
    console.log("  Matching elements (tolerance ".concat(tolerance, "): ").concat(matchingElements, " (").concat(matchPercentage.toFixed(2), "%)"));
    console.log("  Max absolute difference: ".concat(maxDifference.toFixed(6)));
    console.log("  Average absolute difference: ".concat(avgAbsDifference.toFixed(6)));
    if (matchPercentage > 95) {
        console.log("  \u2705 PASS: ".concat(matchPercentage.toFixed(2), "% of outputs match within tolerance"));
    }
    else {
        console.log("  \u274C FAIL: Only ".concat(matchPercentage.toFixed(2), "% of outputs match within tolerance"));
    }
}
function main() {
    return __awaiter(this, void 0, void 0, function () {
        var summaryPath, summaryContent, summary, tracesMetadata, scaledTraces, outputsMetadata, expectedOutputs, inferenceScaling, detectionOutputs, error_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 8, , 9]);
                    console.log("🧠 TypeScript ONNX Spike Detection Inference");
                    console.log("=".repeat(50));
                    summaryPath = "validation/export_summary.json";
                    return [4 /*yield*/, readFile(summaryPath, "utf-8")];
                case 1:
                    summaryContent = _a.sent();
                    summary = JSON.parse(summaryContent);
                    console.log("Configuration loaded:");
                    console.log("  Sample size: ".concat(summary.model_parameters.sample_size));
                    console.log("  Output locations: ".concat(summary.model_parameters.num_output_locs));
                    console.log("  Input scale: ".concat(summary.model_parameters.input_scale));
                    console.log();
                    return [4 /*yield*/, loadMetadata("validation/".concat(summary.files.scaled_traces.metadata))];
                case 2:
                    tracesMetadata = _a.sent();
                    return [4 /*yield*/, loadBinaryData("validation/".concat(summary.files.scaled_traces.binary), tracesMetadata)];
                case 3:
                    scaledTraces = _a.sent();
                    console.log("Scaled traces loaded: shape [".concat(tracesMetadata.shape.join(", "), "]"));
                    return [4 /*yield*/, loadMetadata("validation/".concat(summary.files.model_outputs.metadata))];
                case 4:
                    outputsMetadata = _a.sent();
                    return [4 /*yield*/, loadBinaryData("validation/".concat(summary.files.model_outputs.binary), outputsMetadata)];
                case 5:
                    expectedOutputs = _a.sent();
                    console.log("Expected outputs loaded: shape [".concat(outputsMetadata.shape.join(", "), "]"));
                    console.log();
                    return [4 /*yield*/, runInferenceScaling(scaledTraces, tracesMetadata.shape, summary.model_parameters)];
                case 6:
                    inferenceScaling = _a.sent();
                    console.log();
                    return [4 /*yield*/, runDetection(scaledTraces, tracesMetadata.shape, summary.model_parameters, inferenceScaling)];
                case 7:
                    detectionOutputs = _a.sent();
                    // Step 3: Compare with expected outputs
                    compareOutputs(detectionOutputs, expectedOutputs, outputsMetadata.shape);
                    console.log("\n🎉 Processing complete!");
                    return [3 /*break*/, 9];
                case 8:
                    error_3 = _a.sent();
                    console.error("❌ Error during processing:", error_3);
                    process.exit(1);
                    return [3 /*break*/, 9];
                case 9: return [2 /*return*/];
            }
        });
    });
}
// Run the main function
if (import.meta.url === "file://".concat(process.argv[1])) {
    main().catch(function (error) {
        console.error("Unhandled error:", error);
        process.exit(1);
    });
}
