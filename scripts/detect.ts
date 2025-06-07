/**
 * TypeScript implementation for ONNX-based spike detection inference
 *
 * This script:
 * 1. Loads scaled traces from binary file
 * 2. Runs inference_scaling.onnx on first 1000 frames to calculate scaling factor
 * 3. Processes first 200 frames with detect.onnx to yield first 120 outputs
 * 4. Compares results with expected model outputs
 */

import * as ort from 'onnxruntime-node'
import * as fs from 'fs'
import { promisify } from 'util'

const readFile = promisify(fs.readFile)

interface ModelMetadata {
  shape: number[]
  dtype: string
  byte_size: number
  total_elements: number
  description: string
}

interface ExportSummary {
  files: {
    scaled_traces: {
      binary: string
      metadata: string
      description: string
    }
    model_outputs: {
      binary: string
      metadata: string
      description: string
    }
  }
  model_parameters: {
    sample_size: number
    num_output_locs: number
    input_scale: number
    inference_scaling_numerator: number
    pre_median_frames: number
    buffer_front_sample: number
    buffer_end_sample: number
  }
  processing_notes: string[]
}

class Float16Array {
  private buffer: ArrayBuffer
  private view: DataView
  public length: number

  constructor(data: ArrayBuffer | number) {
    if (typeof data === 'number') {
      this.length = data
      this.buffer = new ArrayBuffer(data * 2) // 2 bytes per float16
    } else {
      this.buffer = data
      this.length = data.byteLength / 2
    }
    this.view = new DataView(this.buffer)
  }

  // Convert float16 to float32 (IEEE 754 implementation)
  private float16ToFloat32(value: number): number {
    const sign = (value >> 15) & 0x1
    const exponent = (value >> 10) & 0x1f
    const mantissa = value & 0x3ff

    if (exponent === 0) {
      if (mantissa === 0) {
        return sign ? -0 : 0
      } else {
        // Subnormal number
        return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024)
      }
    } else if (exponent === 31) {
      return mantissa ? NaN : sign ? -Infinity : Infinity
    } else {
      return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024)
    }
  }

  // Convert float32 to float16 (IEEE 754 implementation)
  static float32ToFloat16(value: number): number {
    if (isNaN(value)) return 0x7e00 // NaN
    if (!isFinite(value)) return value > 0 ? 0x7c00 : 0xfc00 // ±Infinity

    const sign = value < 0 ? 1 : 0
    const absValue = Math.abs(value)

    if (absValue === 0) return sign << 15 // ±0

    // Handle very small numbers (subnormal in float16)
    if (absValue < Math.pow(2, -14)) {
      const mantissa = Math.round(absValue / Math.pow(2, -24))
      return (sign << 15) | mantissa
    }

    // Handle very large numbers (clamp to max float16)
    if (absValue >= 65504) {
      return (sign << 15) | 0x7bff // Max finite float16
    }

    // Normal case
    const exponent = Math.floor(Math.log2(absValue))
    const normalizedMantissa = absValue / Math.pow(2, exponent) - 1
    const mantissaBits = Math.round(normalizedMantissa * 1024)
    const exponentBits = exponent + 15

    return (sign << 15) | (exponentBits << 10) | mantissaBits
  }

  get(index: number): number {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds for Float16Array of length ${this.length}`)
    }
    const uint16Value = this.view.getUint16(index * 2, true) // little endian
    return this.float16ToFloat32(uint16Value)
  }

  toFloat32Array(): Float32Array {
    const result = new Float32Array(this.length)
    for (let i = 0; i < this.length; i++) {
      result[i] = this.get(i)
    }
    return result
  }

  // Get a slice as Float32Array for efficiency
  slice(start: number, end?: number): Float32Array {
    const endIndex = end ?? this.length
    const sliceLength = endIndex - start
    const result = new Float32Array(sliceLength)
    for (let i = 0; i < sliceLength; i++) {
      result[i] = this.get(start + i)
    }
    return result
  }
}

async function loadBinaryData(filePath: string, metadata: ModelMetadata): Promise<Float16Array> {
  console.log(`Loading binary data from ${filePath}...`)
  const buffer = await readFile(filePath)

  // Verify file size matches metadata
  if (buffer.byteLength !== metadata.byte_size) {
    throw new Error(`File size mismatch: expected ${metadata.byte_size}, got ${buffer.byteLength}`)
  }

  return new Float16Array(buffer.buffer)
}

async function loadMetadata(filePath: string): Promise<ModelMetadata> {
  const content = await readFile(filePath, 'utf-8')
  return JSON.parse(content) as ModelMetadata
}

function calculateMedian(arr: Float32Array): number {
  const sorted = Array.from(arr).sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2
  } else {
    return sorted[mid]
  }
}

function calculateQuantile(arr: Float32Array, q: number): number {
  const sorted = Array.from(arr).sort((a, b) => a - b)
  const index = q * (sorted.length - 1)
  const lower = Math.floor(index)
  const upper = Math.ceil(index)

  if (lower === upper) {
    return sorted[lower]
  } else {
    const weight = index - lower
    return sorted[lower] * (1 - weight) + sorted[upper] * weight
  }
}

function reshapeData(data: Float16Array, shape: number[]): Float16Array[] {
  if (shape.length !== 2) {
    throw new Error('Only 2D reshaping supported')
  }

  const [numChannels, timePoints] = shape
  const channels: Float16Array[] = []

  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = new Float32Array(timePoints)
    for (let t = 0; t < timePoints; t++) {
      channelData[t] = data.get(ch * timePoints + t)
    }
    // Convert back to Float16Array for consistency
    const channelBuffer = new ArrayBuffer(timePoints * 2)
    const channelView = new DataView(channelBuffer)
    for (let t = 0; t < timePoints; t++) {
      // Simplified float32 to float16 conversion (may lose precision)
      const value = channelData[t]
      const uint16Value = Math.round(value * 100) / 100 // Simplified
      channelView.setUint16(t * 2, uint16Value, true)
    }
    channels.push(new Float16Array(channelBuffer))
  }

  return channels
}

async function runInferenceScaling(
  scaledTraces: Float16Array,
  shape: number[],
  params: ExportSummary['model_parameters']
): Promise<number> {
  console.log('Running inference scaling calculation...')

  const [numChannels, timePoints] = shape
  const { pre_median_frames, inference_scaling_numerator } = params

  // Extract window for the first pre_median_frames
  const window = new Float32Array(numChannels * pre_median_frames)

  for (let ch = 0; ch < numChannels; ch++) {
    for (let t = 0; t < pre_median_frames; t++) {
      window[ch * pre_median_frames + t] = scaledTraces.get(ch * timePoints + t)
    }
  }

  try {
    // Load and run the inference scaling ONNX model
    const session = await ort.InferenceSession.create('public/models/inference_scaling.onnx')

    // Convert to Uint16Array for float16 tensor using proper IEEE 754 conversion
    const windowUint16 = new Uint16Array(numChannels * pre_median_frames)
    for (let i = 0; i < window.length; i++) {
      windowUint16[i] = Float16Array.float32ToFloat16(window[i])
    }

    const windowTensor = new ort.Tensor('float16', windowUint16, [numChannels, pre_median_frames])
    const numeratorTensor = new ort.Tensor(
      'float32',
      new Float32Array([inference_scaling_numerator]),
      []
    )

    const feeds = {
      window: windowTensor,
      inference_scaling_numerator: numeratorTensor,
    }

    const results = await session.run(feeds)
    const inferenceScaling = results['inference_scaling'].data[0] as number

    console.log(`ONNX inference scaling factor: ${inferenceScaling}`)

    // Also calculate manually for comparison
    const iqrs = new Float32Array(numChannels)
    for (let ch = 0; ch < numChannels; ch++) {
      const channelWindow = window.slice(ch * pre_median_frames, (ch + 1) * pre_median_frames)
      const q25 = calculateQuantile(channelWindow, 0.25)
      const q75 = calculateQuantile(channelWindow, 0.75)
      iqrs[ch] = q75 - q25
    }
    const medianIqr = calculateMedian(iqrs)
    const manualInferenceScaling = medianIqr !== 0 ? inference_scaling_numerator / medianIqr : 1

    console.log(`Manual inference scaling factor: ${manualInferenceScaling}`)
    console.log(`Difference: ${Math.abs(inferenceScaling - manualInferenceScaling)}`)

    return inferenceScaling
  } catch (error) {
    console.warn('ONNX inference scaling failed, calculating manually:', error)

    // Fallback: manual calculation
    const iqrs = new Float32Array(numChannels)

    for (let ch = 0; ch < numChannels; ch++) {
      const channelWindow = window.slice(ch * pre_median_frames, (ch + 1) * pre_median_frames)
      const q25 = calculateQuantile(channelWindow, 0.25)
      const q75 = calculateQuantile(channelWindow, 0.75)
      iqrs[ch] = q75 - q25
    }

    const medianIqr = calculateMedian(iqrs)
    const inferenceScaling = medianIqr !== 0 ? inference_scaling_numerator / medianIqr : 1

    console.log(`Manual inference scaling factor: ${inferenceScaling}`)
    return inferenceScaling
  }
}

async function runDetection(
  scaledTraces: Float16Array,
  shape: number[],
  params: ExportSummary['model_parameters'],
  inferenceScaling: number
): Promise<Float32Array> {
  console.log('Running spike detection...')

  const [numChannels, timePoints] = shape
  const { sample_size, num_output_locs, input_scale } = params

  // Extract first sample_size frames for processing
  const windowData = new Float32Array(numChannels * sample_size)

  for (let ch = 0; ch < numChannels; ch++) {
    for (let t = 0; t < sample_size; t++) {
      windowData[ch * sample_size + t] = scaledTraces.get(ch * timePoints + t)
    }
  }

  // Calculate median for each channel and subtract it
  const processedWindow = new Float32Array(numChannels * sample_size)

  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = windowData.slice(ch * sample_size, (ch + 1) * sample_size)
    const median = calculateMedian(channelData)

    for (let t = 0; t < sample_size; t++) {
      const baselineSubtracted = channelData[t] - median
      processedWindow[ch * sample_size + t] = baselineSubtracted * input_scale * inferenceScaling
    }
  }

  try {
    // Load and run the detection ONNX model
    const session = await ort.InferenceSession.create('public/models/detect.onnx')

    // Convert to Uint16Array for float16 tensor using proper IEEE 754 conversion
    const processedWindowUint16 = new Uint16Array(numChannels * sample_size)
    for (let i = 0; i < processedWindow.length; i++) {
      processedWindowUint16[i] = Float16Array.float32ToFloat16(processedWindow[i])
    }

    // Reshape for model input: [numChannels, 1, sample_size]
    const inputTensor = new ort.Tensor('float16', processedWindowUint16, [
      numChannels,
      1,
      sample_size,
    ])

    const feeds = { input: inputTensor }
    const results = await session.run(feeds)

    const outputs = results['output'].data as Float32Array
    console.log(`Detection completed. Output shape: [${numChannels}, 1, ${num_output_locs}]`)
    console.log(
      `First few outputs: [${Array.from(outputs.slice(0, 10))
        .map((x) => x.toFixed(4))
        .join(', ')}...]`
    )

    return outputs
  } catch (error) {
    console.error('ONNX detection failed:', error)
    throw error
  }
}

function compareOutputs(predicted: Float32Array, expected: Float16Array, shape: number[]): void {
  console.log('\nComparing predicted vs expected outputs...')

  const [numChannels, numOutputLocs] = [shape[0], 120] // First 120 outputs
  const tolerance = 1e-2

  let totalElements = 0
  let matchingElements = 0
  let maxDifference = 0
  let totalAbsoluteDifference = 0

  for (let ch = 0; ch < numChannels; ch++) {
    for (let t = 0; t < numOutputLocs; t++) {
      const predIdx = ch * numOutputLocs + t
      const expIdx = ch * shape[1] + t // expected has full time dimension

      if (predIdx < predicted.length && expIdx < expected.length) {
        const predValue = predicted[predIdx]
        const expValue = expected.get(expIdx)
        const difference = Math.abs(predValue - expValue)

        totalElements++
        totalAbsoluteDifference += difference

        if (difference <= tolerance) {
          matchingElements++
        }

        if (difference > maxDifference) {
          maxDifference = difference
        }

        // Log first few comparisons for debugging
        if (totalElements <= 5) {
          console.log(
            `  Element ${totalElements}: predicted=${predValue.toFixed(
              4
            )}, expected=${expValue.toFixed(4)}, diff=${difference.toFixed(4)}`
          )
        }
      }
    }
  }

  const matchPercentage = (matchingElements / totalElements) * 100
  const avgAbsDifference = totalAbsoluteDifference / totalElements

  console.log(`\nComparison Results:`)
  console.log(`  Total elements compared: ${totalElements}`)
  console.log(
    `  Matching elements (tolerance ${tolerance}): ${matchingElements} (${matchPercentage.toFixed(
      2
    )}%)`
  )
  console.log(`  Max absolute difference: ${maxDifference.toFixed(6)}`)
  console.log(`  Average absolute difference: ${avgAbsDifference.toFixed(6)}`)

  if (matchPercentage > 95) {
    console.log(`  ✅ PASS: ${matchPercentage.toFixed(2)}% of outputs match within tolerance`)
  } else {
    console.log(`  ❌ FAIL: Only ${matchPercentage.toFixed(2)}% of outputs match within tolerance`)
  }
}

async function main(): Promise<void> {
  try {
    console.log('🧠 TypeScript ONNX Spike Detection Inference')
    console.log('='.repeat(50))

    // Load configuration
    const summaryPath = 'public/models/export_summary.json'
    const summaryContent = await readFile(summaryPath, 'utf-8')
    const summary: ExportSummary = JSON.parse(summaryContent)

    console.log('Configuration loaded:')
    console.log(`  Sample size: ${summary.model_parameters.sample_size}`)
    console.log(`  Output locations: ${summary.model_parameters.num_output_locs}`)
    console.log(`  Input scale: ${summary.model_parameters.input_scale}`)
    console.log()

    // Load scaled traces
    const tracesMetadata = await loadMetadata(
      `public/models/${summary.files.scaled_traces.metadata}`
    )
    const scaledTraces = await loadBinaryData(
      `public/models/${summary.files.scaled_traces.binary}`,
      tracesMetadata
    )

    console.log(`Scaled traces loaded: shape [${tracesMetadata.shape.join(', ')}]`)

    // Load expected outputs for comparison
    const outputsMetadata = await loadMetadata(
      `public/models/${summary.files.model_outputs.metadata}`
    )
    const expectedOutputs = await loadBinaryData(
      `public/models/${summary.files.model_outputs.binary}`,
      outputsMetadata
    )

    console.log(`Expected outputs loaded: shape [${outputsMetadata.shape.join(', ')}]`)
    console.log()

    // Step 1: Calculate inference scaling
    const inferenceScaling = await runInferenceScaling(
      scaledTraces,
      tracesMetadata.shape,
      summary.model_parameters
    )

    console.log()

    // Step 2: Run detection on first window
    const detectionOutputs = await runDetection(
      scaledTraces,
      tracesMetadata.shape,
      summary.model_parameters,
      inferenceScaling
    )

    // Step 3: Compare with expected outputs
    compareOutputs(detectionOutputs, expectedOutputs, outputsMetadata.shape)

    console.log('\n🎉 Processing complete!')
  } catch (error) {
    console.error('❌ Error during processing:', error)
    process.exit(1)
  }
}

// Run the main function
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error('Unhandled error:', error)
    process.exit(1)
  })
}
