/**
 * Browser web worker for running rt-sort models via onnx
 */

import { InferenceSession, Tensor, env } from 'onnxruntime-web'
import h5wasm from 'h5wasm'

interface PredictionMessage {
  type: 'start' | 'openFile' | 'runWithH5'
  modelsURL?: string
  useGPU?: boolean
  h5File?: File
  file?: File
}

interface H5FileParameters {
  samplingRate: number
  numChannels: number
  numSamples: number
  duration: number
  gain: number
  lsb: number
  hpf?: number
}

interface H5DataSet {
  type: string
  value: any
  shape: number[]
  keys(): string[]
  slice(ranges: any[][]): any
}

interface H5Group {
  type: string
  keys(): string[]
  get(path: string): H5DataSet | H5Group
}

interface H5File {
  keys(): string[]
  get(path: string): H5DataSet | H5Group
  close(): void
}

/**
 * Check for GPU availability and determine the best execution provider
 */
async function getBestExecutionProvider(useGPU: boolean = false): Promise<string[]> {
  if (!useGPU) {
    return ['wasm']
  }

  // Check for WebGPU support first (best performance)
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter()
      if (adapter) {
        console.log('🚀 WebGPU available, using webgpu execution provider')
        return ['webgpu', 'wasm'] // fallback to wasm if webgpu fails
      }
    } catch (error) {
      console.warn('WebGPU adapter request failed:', error)
    }
  }

  // Check for WebGL support (fallback GPU option)
  if (typeof document !== 'undefined') {
    try {
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl')
      if (gl) {
        console.log('🎮 WebGL available, using webgl execution provider')
        return ['webgl', 'wasm'] // fallback to wasm if webgl fails
      }
    } catch (error) {
      console.warn('WebGL context creation failed:', error)
    }
  }

  console.log('⚠️ No GPU support detected, falling back to CPU (wasm)')
  return ['wasm']
}
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
  inferenceScalingNumerator: number = 12.6,
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
 * Run detection model on windowed data with benchmarking
 * @param inferenceScaling - The computed inference scaling value
 * @param windowData - The full window data as Float16Array
 * @param detectSession - The ONNX detection session
 * @param numChannels - Number of channels (942)
 * @param totalSamples - Total samples available (2000)
 * @returns Generator yielding detection outputs and timing for each window
 */
async function* runDetectionModelWithBenchmark(
  inferenceScaling: number,
  windowData: Float16Array,
  detectSession: InferenceSession,
  numChannels: number = 942,
  totalSamples: number = 2000,
): AsyncGenerator<{ output: Float32Array; duration: number }, void, unknown> {
  const sampleSize = 200
  const numOutputLocs = 120
  const inputScale = 0.15887516

  // Process windows
  for (let startFrame = 0; startFrame <= totalSamples - sampleSize; startFrame += numOutputLocs) {
    const startTime = performance.now()

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

      // Send progress update to the main thread
      self.postMessage({
        type: 'processingProgress',
        message: 'Detecting spikes...',
        countFinished: startFrame,
        totalToProcess: totalSamples - sampleSize,
      })
    }

    // Create ONNX tensor with shape [numChannels, 1, sampleSize]
    // Convert Float16Array to Uint16Array for ONNX float16 tensor
    const uint16Data = new Uint16Array(processedData.buffer)
    const inputTensor = new Tensor('float16', uint16Data, [numChannels, 1, sampleSize])

    // Run detection model
    const results = await detectSession.run({ input: inputTensor })

    const endTime = performance.now()
    const duration = endTime - startTime

    yield {
      output: results.output.data as Float32Array,
      duration,
    }
  }
}

/**
 * Extract file parameters from Maxwell h5 file
 */
async function extractH5FileParameters(h5File: File): Promise<H5FileParameters> {
  console.log('🔄 Extracting parameters from h5 file...')

  // Initialize h5wasm
  const Module = await h5wasm.ready
  const { FS } = Module

  try {
    // Mount the file to the filesystem
    if (!FS.analyzePath('/work').exists) {
      FS.mkdir('/work')
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, '/work')

    // Open the h5 file
    const h5 = new h5wasm.File(`/work/${h5File.name}`, 'r') as H5File

    // Get signal data shape
    const sig = h5.get('sig') as H5DataSet
    const [numChannels, numSamples] = sig.shape

    // Get settings
    const settings = h5.get('settings') as H5Group
    const gain = (settings.get('gain') as H5DataSet).value[0]
    const lsb = (settings.get('lsb') as H5DataSet).value[0]

    // Try to get high-pass filter setting if available
    let hpf: number | undefined
    try {
      hpf = (settings.get('hpf') as H5DataSet).value[0]
    } catch (e) {
      console.log('HPF setting not found, will use default')
    }

    // Calculate sampling rate (typically 20kHz for Maxwell)
    // We can try to get this from the recording or use default
    let samplingRate = 20000.0 // Default for Maxwell

    // Try to get actual sampling rate if available in the file
    try {
      // Some Maxwell files may have this information
      if (h5.keys().includes('time')) {
        const timeData = h5.get('time') as H5DataSet
        if (timeData.shape[0] > 1) {
          const timeStep = timeData.value[1] - timeData.value[0]
          samplingRate = 1.0 / timeStep
        }
      }
    } catch (e) {
      console.log('Could not determine sampling rate from file, using default 20kHz')
    }

    const duration = numSamples / samplingRate

    const parameters: H5FileParameters = {
      samplingRate,
      numChannels,
      numSamples,
      duration,
      gain,
      lsb,
      hpf,
    }

    console.log('📊 H5 File Parameters:', parameters)

    // Close the file for now
    h5.close()
    FS.unmount('/work')

    return parameters
  } catch (error) {
    console.error('❌ Error extracting h5 file parameters:', error)

    // Clean up on error
    try {
      FS.unmount('/work')
    } catch (e) {
      // Ignore unmount errors
    }

    throw error
  }
}

/**
 * Stream raw uint16 frames from h5 file
 */
async function* streamRawFramesFromH5(
  h5File: File,
  startFrame: number = 0,
  numFrames: number = 1000,
): AsyncGenerator<{ frames: Uint16Array; actualFrames: number }, void, unknown> {
  console.log(`🔄 Streaming ${numFrames} frames starting at ${startFrame}...`)

  // Initialize h5wasm
  const Module = await h5wasm.ready
  const { FS } = Module

  try {
    // Mount the file to the filesystem
    if (!FS.analyzePath('/work').exists) {
      FS.mkdir('/work')
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, '/work')

    // Open the h5 file
    const h5 = new h5wasm.File(`/work/${h5File.name}`, 'r') as H5File
    const sig = h5.get('sig') as H5DataSet
    const [numChannels, totalSamples] = sig.shape

    // Ensure we don't read beyond the file
    const endFrame = Math.min(startFrame + numFrames, totalSamples)
    const actualFrames = endFrame - startFrame

    if (actualFrames <= 0) {
      console.log('No frames to read')
      return
    }

    // Read the data slice: [all channels, startFrame:endFrame]
    const frameData = sig.slice([
      [0, numChannels],
      [startFrame, endFrame],
    ])

    // Convert to Uint16Array (Maxwell data is stored as uint16)
    const frames = new Uint16Array(frameData.flat())

    console.log(`✅ Read ${actualFrames} frames for ${numChannels} channels`)

    yield { frames, actualFrames }

    // Close the file
    h5.close()
    FS.unmount('/work')
  } catch (error) {
    console.error('❌ Error streaming frames from h5 file:', error)

    // Clean up on error
    try {
      FS.unmount('/work')
    } catch (e) {
      // Ignore unmount errors
    }

    throw error
  }
}

/**
 * Scale raw uint16 frames to physical values and then to scaled traces
 * @param rawFrames - Raw uint16 frames from h5 file
 * @param parameters - H5 file parameters (gain, lsb, etc.)
 * @param numChannels - Number of channels
 * @param numFrames - Number of frames
 * @returns Scaled traces as Float16Array
 */
function scaleRawFramesToFloat16(
  rawFrames: Uint16Array,
  parameters: H5FileParameters,
  numChannels: number,
  numFrames: number,
): Float16Array {
  console.log(`🔄 Scaling ${numChannels} channels x ${numFrames} frames...`)

  // Convert raw uint16 to physical values (microvolts)
  // Formula: physical_value = raw * lsb * 1e6 * gain
  const scalingFactor = parameters.lsb * 1e6 * parameters.gain

  // Create Float16Array for scaled traces
  const scaledTraces = new Float16Array(numChannels * numFrames)

  for (let i = 0; i < rawFrames.length; i++) {
    // Convert to physical value and then to float16
    const physicalValue = rawFrames[i] * scalingFactor
    scaledTraces[i] = physicalValue
  }

  console.log(`✅ Scaling completed`)
  return scaledTraces
}

/**
 * Re-implement calculateInferenceScaling to work with h5 streaming
 */
async function calculateInferenceScalingFromH5(
  h5File: File,
  parameters: H5FileParameters,
  preMedianFrames: number = 1000,
  inferenceScalingNumerator: number = 12.6,
): Promise<number> {
  console.log('🔄 Calculating inference scaling from h5 file...')

  // Stream the first preMedianFrames from the h5 file
  for await (const { frames, actualFrames } of streamRawFramesFromH5(h5File, 0, preMedianFrames)) {
    // Scale the raw frames to float16
    const scaledTraces = scaleRawFramesToFloat16(
      frames,
      parameters,
      parameters.numChannels,
      actualFrames,
    )

    // Use existing calculateInferenceScaling logic
    return calculateInferenceScaling(
      scaledTraces,
      parameters.numChannels,
      actualFrames,
      actualFrames,
      inferenceScalingNumerator,
    )
  }

  throw new Error('Failed to read frames for inference scaling calculation')
}

/**
 * Re-implement runDetectionModelWithBenchmark to work with h5 streaming
 */
async function* runDetectionModelWithBenchmarkFromH5(
  h5File: File,
  parameters: H5FileParameters,
  inferenceScaling: number,
  detectSession: InferenceSession,
): AsyncGenerator<{ output: Float32Array; duration: number }, void, unknown> {
  const sampleSize = 200
  const numOutputLocs = 120
  const inputScale = 0.15887516

  const numChannels = parameters.numChannels
  const totalSamples = parameters.numSamples

  // Process windows
  for (let startFrame = 0; startFrame <= totalSamples - sampleSize; startFrame += numOutputLocs) {
    const startTime = performance.now()

    // Stream this window from the h5 file
    for await (const { frames, actualFrames } of streamRawFramesFromH5(
      h5File,
      startFrame,
      sampleSize,
    )) {
      // Scale the raw frames
      const windowData = scaleRawFramesToFloat16(frames, parameters, numChannels, actualFrames)

      // Extract and process window data for each channel
      const processedData = new Float16Array(numChannels * sampleSize)

      for (let channel = 0; channel < numChannels; channel++) {
        // Extract channel data for this window
        const channelStart = channel * actualFrames
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
      const uint16Data = new Uint16Array(processedData.buffer)
      const inputTensor = new Tensor('float16', uint16Data, [numChannels, 1, sampleSize])

      // Run detection model
      const results = await detectSession.run({ input: inputTensor })

      const endTime = performance.now()
      const duration = endTime - startTime

      // Send progress update to the main thread
      self.postMessage({
        type: 'processingProgress',
        message: 'Detecting spikes...',
        countFinished: startFrame,
        totalToProcess: totalSamples - sampleSize,
      })

      yield {
        output: results.output.data as Float32Array,
        duration,
      }
    }
  }
}

self.addEventListener('message', async function (event: MessageEvent<PredictionMessage>) {
  if (event.data.type === 'start') {
    console.log('Received start message')
    console.log('Model URL:', event.data.modelsURL)
    console.log('Use GPU:', event.data.useGPU ?? false)

    try {
      // Determine execution providers
      const executionProviders = await getBestExecutionProvider(event.data.useGPU ?? false)
      console.log('Selected execution providers:', executionProviders)

      // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
      const numThreads = navigator.hardwareConcurrency - 1
      console.log(`Number of threads: ${numThreads}`)
      env.wasm.numThreads = numThreads
      env.wasm.proxy = true

      const options: InferenceSession.SessionOptions = {
        executionProviders: executionProviders,
        executionMode: 'parallel',
      }

      const detectSession = await InferenceSession.create(
        `${event.data.modelsURL}/detect-mea.onnx`,
        options,
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

      console.log('Starting inference scaling computation...')
      const inferenceScalingValue = calculateInferenceScaling(
        scaledTraces,
        numChannels,
        totalSamples,
      )
      console.log(`✅ Inference scaling computed successfully: ${inferenceScalingValue}`)

      console.log('🔄 Running detection model...')
      const detectionResults: Float32Array[] = []
      const benchmarks: number[] = []

      for await (const { output, duration } of runDetectionModelWithBenchmark(
        inferenceScalingValue,
        scaledTraces,
        detectSession,
        numChannels,
        totalSamples,
      )) {
        detectionResults.push(output)
        benchmarks.push(duration)
      }

      console.log(`✅ Detection completed. Generated ${detectionResults.length} windows`)

      // Calculate benchmark statistics
      const avgTime = benchmarks.reduce((a, b) => a + b, 0) / benchmarks.length
      const minTime = Math.min(...benchmarks)
      const maxTime = Math.max(...benchmarks)

      console.log(`⏱️ Inference benchmarks (per 200-sample window):`)
      console.log(`   Average: ${avgTime.toFixed(2)}ms`)
      console.log(`   Min: ${minTime.toFixed(2)}ms`)
      console.log(`   Max: ${maxTime.toFixed(2)}ms`)
      console.log(`   Provider: ${executionProviders[0]}`)

      // Send results back to main thread
      self.postMessage({
        type: 'result',
        result: {
          inferenceScaling: inferenceScalingValue,
          detectionOutputs: detectionResults,
          benchmarks: {
            avgTime,
            minTime,
            maxTime,
            provider: executionProviders[0],
            totalWindows: benchmarks.length,
          },
        },
      })
    } catch (error) {
      console.error('❌ Failed to run detection:', error)
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : String(error),
      })
    }
  } else if (event.data.type === 'openFile' && event.data.file) {
    console.log('Received openFile message')
    console.log('H5 File:', event.data.file.name)

    try {
      // Extract parameters from h5 file
      const parameters = await extractH5FileParameters(event.data.file)

      // Send parameters back to main thread
      self.postMessage({
        type: 'fileParameters',
        parameters,
      })
    } catch (error) {
      console.error('❌ Failed to extract file parameters:', error)
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : String(error),
      })
    }
  } else if (event.data.type === 'runWithH5' && event.data.file) {
    console.log('Received runWithH5 message')
    console.log('Model URL:', event.data.modelsURL)
    console.log('H5 File:', event.data.file.name)
    console.log('Use GPU:', event.data.useGPU ?? false)

    try {
      // First extract parameters from h5 file
      const parameters = await extractH5FileParameters(event.data.file)
      console.log('📊 H5 File Parameters:', parameters)

      // Determine execution providers
      const executionProviders = await getBestExecutionProvider(event.data.useGPU ?? false)
      console.log('Selected execution providers:', executionProviders)

      // Initialize ONNX Runtime
      const numThreads = navigator.hardwareConcurrency - 1
      console.log(`Number of threads: ${numThreads}`)
      env.wasm.numThreads = numThreads
      env.wasm.proxy = true

      const options: InferenceSession.SessionOptions = {
        executionProviders: executionProviders,
        executionMode: 'parallel',
      }

      const detectSession = await InferenceSession.create(
        `${event.data.modelsURL}/detect-mea.onnx`,
        options,
      )
      console.log('Detection model loaded:', detectSession.outputNames)

      // Calculate inference scaling from h5 file
      console.log('🔄 Calculating inference scaling from h5 file...')
      const inferenceScalingValue = await calculateInferenceScalingFromH5(
        event.data.file,
        parameters,
      )
      console.log(`✅ Inference scaling computed: ${inferenceScalingValue}`)

      // Run detection model with h5 streaming
      console.log('🔄 Running detection model with h5 streaming...')
      const detectionResults: Float32Array[] = []
      const benchmarks: number[] = []

      for await (const { output, duration } of runDetectionModelWithBenchmarkFromH5(
        event.data.file,
        parameters,
        inferenceScalingValue,
        detectSession,
      )) {
        detectionResults.push(output)
        benchmarks.push(duration)
      }

      console.log(`✅ Detection completed. Generated ${detectionResults.length} windows`)

      // Calculate benchmark statistics
      const avgTime = benchmarks.reduce((a, b) => a + b, 0) / benchmarks.length
      const minTime = Math.min(...benchmarks)
      const maxTime = Math.max(...benchmarks)

      console.log(`⏱️ Inference benchmarks (per 200-sample window):`)
      console.log(`   Average: ${avgTime.toFixed(2)}ms`)
      console.log(`   Min: ${minTime.toFixed(2)}ms`)
      console.log(`   Max: ${maxTime.toFixed(2)}ms`)
      console.log(`   Provider: ${executionProviders[0]}`)

      // Send results back to main thread
      self.postMessage({
        type: 'result',
        result: {
          inferenceScaling: inferenceScalingValue,
          detectionOutputs: detectionResults,
          parameters,
          benchmarks: {
            avgTime,
            minTime,
            maxTime,
            provider: executionProviders[0],
            totalWindows: benchmarks.length,
          },
        },
      })
    } catch (error) {
      console.error('❌ Failed to run detection with h5 file:', error)
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
})
