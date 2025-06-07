/**
 * Browser web worker for running rt-sort models via onnx
 */

import { InferenceSession, Tensor, env } from 'onnxruntime-web'

interface PredictionMessage {
  type: 'start'
  modelsURL: string
}

/**
 * Calculate linear interpolation for quantiles (matches PyTorch behavior)
 */
function linearInterpolate(sortedData: Float32Array, quantile: number): number {
  const n = sortedData.length
  const index = quantile * (n - 1)
  const lowerIndex = Math.floor(index)
  const upperIndex = Math.ceil(index)

  if (lowerIndex === upperIndex) {
    return sortedData[lowerIndex]
  }

  const fraction = index - lowerIndex
  return sortedData[lowerIndex] * (1 - fraction) + sortedData[upperIndex] * fraction
}

/**
 * Calculate median using the same method as PyTorch
 */
function calculateMedian(data: Float32Array): number {
  const sorted = new Float32Array(data)
  sorted.sort()

  const n = sorted.length
  if (n % 2 === 0) {
    // For even length, return average of two middle elements
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
  } else {
    // For odd length, return middle element
    return sorted[Math.floor(n / 2)]
  }
}

/**
 * Calculate inference scaling based on initial window IQR analysis
 * Uses the exact same logic as the Python PyTorch implementation
 * @param scaledTraces - The Float16Array containing scaled traces [numChannels * totalSamples]
 * @param numChannels - Number of channels (942)
 * @param totalSamples - Total samples available (2000)
 * @param preMedianFrames - Number of frames to use for initial window (1000)
 * @param inferenceScalingNumerator - Numerator value for scaling calculation (12.6)
 * @returns The computed inference scaling value
 */
function calculateInferenceScaling(
  scaledTraces: Float16Array,
  numChannels: number,
  totalSamples: number,
  preMedianFrames: number = 1000,
  inferenceScalingNumerator: number = 12.6
): number {
  // Extract initial window for each channel [numChannels, preMedianFrames]
  const window = new Float32Array(numChannels * preMedianFrames)

  for (let channel = 0; channel < numChannels; channel++) {
    const channelStart = channel * totalSamples
    for (let sample = 0; sample < preMedianFrames; sample++) {
      // Convert Float16 to Float32 as per Python code
      window[channel * preMedianFrames + sample] = scaledTraces[channelStart + sample]
    }
  }

  // Calculate IQR for each channel
  const iqrs = new Float32Array(numChannels)

  for (let channel = 0; channel < numChannels; channel++) {
    // Extract channel data
    const channelData = new Float32Array(preMedianFrames)
    for (let i = 0; i < preMedianFrames; i++) {
      channelData[i] = window[channel * preMedianFrames + i]
    }

    // Sort for quantile calculation
    channelData.sort()

    // Calculate 25th and 75th percentiles using linear interpolation (matches PyTorch)
    const q25 = linearInterpolate(channelData, 0.25)
    const q75 = linearInterpolate(channelData, 0.75)

    // IQR = Q75 - Q25
    iqrs[channel] = q75 - q25
  }

  // Calculate median of IQRs using PyTorch-compatible method
  const medianIqr = calculateMedian(iqrs)

  // Calculate inference scaling
  const inferenceScaling = medianIqr !== 0 ? inferenceScalingNumerator / medianIqr : 1

  console.log(`📊 Median IQR: ${medianIqr}, Inference Scaling: ${inferenceScaling}`)

  return inferenceScaling
}

/**
 * Run detection model on windowed data
 * @param inferenceScaling - The computed inference scaling value
 * @param windowData - The full window data as Float16Array
 * @param detectSession - The ONNX detection session
 * @param numChannels - Number of channels (942)
 * @param totalSamples - Total samples available (2000)
 * @returns Generator yielding detection outputs for each window
 */
async function* runDetectionModel(
  inferenceScaling: number,
  windowData: Float16Array,
  detectSession: InferenceSession,
  numChannels: number = 942,
  totalSamples: number = 2000
): AsyncGenerator<Float32Array, void, unknown> {
  const sampleSize = 200
  const numOutputLocs = 120
  const inputScale = 0.15887516

  // Process windows
  for (let startFrame = 0; startFrame <= totalSamples - sampleSize; startFrame += numOutputLocs) {
    // Extract and process window data for each channel
    const processedData = new Float16Array(numChannels * sampleSize)

    for (let channel = 0; channel < numChannels; channel++) {
      // Extract channel data for this window
      const channelStart = channel * totalSamples + startFrame
      const channelData = windowData.slice(channelStart, channelStart + sampleSize)

      // Calculate median for baseline correction
      const sorted = Array.from(channelData).sort((a, b) => a - b)
      const median = sorted[Math.floor(sorted.length / 2)]

      // Subtract median and apply scaling
      for (let i = 0; i < sampleSize; i++) {
        processedData[channel * sampleSize + i] =
          (channelData[i] - median) * inputScale * inferenceScaling
      }
    }

    // Create ONNX tensor with shape [numChannels, 1, sampleSize]
    // Convert Float16Array to Uint16Array for ONNX float16 tensor
    const uint16Data = new Uint16Array(processedData.buffer)
    const inputTensor = new Tensor('float16', uint16Data, [numChannels, 1, sampleSize])

    // Run detection model
    const results = await detectSession.run({ input: inputTensor })

    yield results.output.data as Float32Array
  }
}

self.addEventListener('message', async function (event: MessageEvent<PredictionMessage>) {
  if (event.data.type === 'start') {
    console.log('Received start message')
    console.log('Model URL:', event.data.modelsURL)

    // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
    const numThreads = navigator.hardwareConcurrency - 1
    console.log(`Number of threads: ${numThreads}`)
    env.wasm.numThreads = numThreads
    env.wasm.proxy = true
    const options: InferenceSession.SessionOptions = {
      executionProviders: ['wasm'], // alias of 'cpu'
      executionMode: 'parallel',
    }

    const detectSession = await InferenceSession.create(
      `${event.data.modelsURL}/detect-mea.onnx`,
      options
    )
    console.log('Detection model output names', detectSession.outputNames)

    // Load the window from scaled_traces.bin
    console.log('🔄 Fetching scaled_traces.bin...')
    const response = await fetch(`${event.data.modelsURL}/scaled_traces.bin`)
    if (!response.ok) {
      throw new Error(`Failed to fetch scaled_traces.bin: ${response.statusText}`)
    }
    const arrayBuffer = await response.arrayBuffer()
    const scaledTraces = new Float16Array(arrayBuffer)
    const numChannels = 942
    const totalSamples = 2000

    try {
      console.log('Starting inference scaling computation...')
      const inferenceScalingValue = calculateInferenceScaling(
        scaledTraces,
        numChannels,
        totalSamples
      )
      console.log(`✅ Inference scaling computed successfully: ${inferenceScalingValue}`)

      console.log('🔄 Running detection model...')
      const detectionResults: Float32Array[] = []
      for await (const output of runDetectionModel(
        inferenceScalingValue,
        scaledTraces,
        detectSession,
        numChannels,
        totalSamples
      )) {
        detectionResults.push(output)
      }
      console.log(`✅ Detection completed. Generated ${detectionResults.length} windows`)

      // Send results back to main thread
      self.postMessage({
        type: 'result',
        result: {
          inferenceScaling: inferenceScalingValue,
          detectionOutputs: detectionResults,
        },
      })
    } catch (error) {
      console.error('❌ Failed to run detection:', error)
    }
  }
})
