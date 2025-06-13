/**
 * Browser web worker for running rt-sort models via onnx
 */

import { InferenceSession, Tensor, env } from 'onnxruntime-web'
import h5wasm from 'h5wasm'

interface PredictionMessage {
  type: 'open' | 'run' | 'stop'
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
  storageLayout: 'row-major' | 'column-major'
  chunkSize?: number[]
  isChunked: boolean
  fileFormat?: 'maxwell' | 'nwb' | 'unknown'
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

/**
 * Linear interpolation for quantile calculation
 * Matches PyTorch's quantile calculation method
 * @param sortedData - Sorted Float32Array data
 * @param quantile - Quantile to calculate (0.0 to 1.0)
 * @returns Interpolated value at the given quantile
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
 * Detect HDF5 dataset storage layout and chunking information
 */
function detectStorageLayout(
  dataset: H5DataSet,
  filePath?: string,
): {
  layout: 'row-major' | 'column-major'
  chunkSize?: number[]
  isChunked: boolean
  fileFormat?: 'maxwell' | 'nwb' | 'unknown'
} {
  try {
    // Check if dataset has chunking information
    // h5wasm may expose this through dataset properties
    const datasetInfo = dataset as any

    // Check for chunk information (this is implementation-specific to h5wasm)
    let chunkSize: number[] | undefined
    let isChunked = false

    if (datasetInfo.chunks) {
      chunkSize = datasetInfo.chunks
      isChunked = true
    } else if (datasetInfo.chunk_size) {
      chunkSize = datasetInfo.chunk_size
      isChunked = true
    }

    // Detect file format for better heuristics
    const isNWB =
      filePath?.toLowerCase().includes('nwb') || filePath?.toLowerCase().endsWith('.nwb')
    const isMaxwell =
      filePath?.toLowerCase().includes('maxwell') || filePath?.toLowerCase().includes('mea')

    let fileFormat: 'maxwell' | 'nwb' | 'unknown' = 'unknown'
    if (isNWB) fileFormat = 'nwb'
    else if (isMaxwell) fileFormat = 'maxwell'

    // Determine layout based on chunk pattern or shape
    // For MEA data: [channels, samples] suggests row-major (C-order)
    // [samples, channels] suggests column-major (Fortran-order)
    const [dim0, dim1] = dataset.shape

    // Heuristic: if first dimension is much smaller than second,
    // likely channels x samples (row-major)
    // Maxwell typically stores as [channels, samples]
    let layout: 'row-major' | 'column-major'

    if (chunkSize) {
      const [chunkDim0, chunkDim1] = chunkSize
      layout = chunkDim1 > chunkDim0 ? 'row-major' : 'column-major'
    } else {
      // Format-specific heuristics
      if (fileFormat === 'nwb') {
        // NWB typically uses (time, channels), so if dim0 >> dim1, likely column-major
        layout = dim0 > dim1 ? 'column-major' : 'row-major'
      } else if (fileFormat === 'maxwell') {
        // Maxwell typically uses (channels, time), so if dim0 < dim1, likely row-major
        layout = dim0 < dim1 ? 'row-major' : 'column-major'
      } else {
        // Default heuristic
        layout = dim0 < dim1 ? 'row-major' : 'column-major'
      }
    }

    return { layout, chunkSize, isChunked, fileFormat }
  } catch (error) {
    console.warn('Could not detect storage layout, assuming row-major:', error)
    return { layout: 'row-major', isChunked: false, fileFormat: 'unknown' }
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

    // Get signal data shape and storage info
    const sig = h5.get('sig') as H5DataSet
    const [numChannels, numSamples] = sig.shape
    const storageInfo = detectStorageLayout(sig, h5File.name)

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
      storageLayout: storageInfo.layout,
      chunkSize: storageInfo.chunkSize,
      isChunked: storageInfo.isChunked,
      fileFormat: storageInfo.fileFormat,
    }

    console.log('📊 H5 File Parameters:', parameters)
    console.log(
      `💾 Storage Layout: ${storageInfo.layout}, Chunked: ${storageInfo.isChunked}, Format: ${storageInfo.fileFormat}`,
    )
    if (storageInfo.chunkSize) {
      console.log(`📦 Chunk Size: [${storageInfo.chunkSize.join(', ')}]`)
    }

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
  // Convert raw uint16 to physical values (microvolts)
  // REMIND: lsb must imply the 1e6...spelunk into:
  // https://github.com/SpikeInterface/spikeinterface/blob/fb3bda89828134800938f33e7d7d938872f01fbd/src/spikeinterface/extractors/neoextractors/neobaseextractor.py#L249
  const scalingFactor = parameters.lsb * 1e6

  // Create Float16Array for scaled traces
  const scaledTraces = new Float16Array(numChannels * numFrames)

  for (let i = 0; i < rawFrames.length; i++) {
    // Convert to physical value and then to float16
    const physicalValue = rawFrames[i] * scalingFactor
    scaledTraces[i] = physicalValue
  }

  return scaledTraces
}

/**
 * Re-implement calculateInferenceScaling to work with h5 streaming and return validation data
 */
async function calculateInferenceScalingFromH5WithValidation(
  h5File: File,
  parameters: H5FileParameters,
  preMedianFrames: number = 1000,
  inferenceScalingNumerator: number = 12.6,
): Promise<{ inferenceScaling: number; rawFrames: Uint16Array; scaledTraces: Float16Array }> {
  console.log('🔄 Calculating inference scaling from h5 file with validation...')

  // Initialize h5wasm and handle mounting here
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
    const actualFrames = Math.min(preMedianFrames, totalSamples)

    if (actualFrames <= 0) {
      throw new Error('No frames to read for inference scaling calculation')
    }

    // Read the data slice: [all channels, 0:actualFrames]
    const frameData = sig.slice([
      [0, numChannels],
      [0, actualFrames],
    ])

    // Convert to Uint16Array
    let frames: Uint16Array
    if (frameData instanceof Uint16Array) {
      frames = frameData
    } else if (ArrayBuffer.isView(frameData)) {
      frames = new Uint16Array(frameData.buffer, frameData.byteOffset, frameData.byteLength / 2)
    } else if (Array.isArray(frameData)) {
      const flatData = frameData.flat()
      frames = new Uint16Array(flatData)
    } else {
      frames = new Uint16Array(frameData)
    }

    console.log(`✅ Read ${actualFrames} frames for ${numChannels} channels`)

    // Scale the raw frames to float16
    const scaledTraces = scaleRawFramesToFloat16(
      frames,
      parameters,
      parameters.numChannels,
      actualFrames,
    )

    // Use existing calculateInferenceScaling logic
    const inferenceScaling = calculateInferenceScaling(
      scaledTraces,
      parameters.numChannels,
      actualFrames,
      actualFrames,
      inferenceScalingNumerator,
    )

    // Close the file and unmount
    h5.close()
    FS.unmount('/work')

    return { inferenceScaling, rawFrames: frames, scaledTraces }
  } catch (error) {
    console.error('❌ Error calculating inference scaling from h5 file:', error)

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
 * Optimized window reading based on storage layout
 */
function readWindowOptimized(
  sig: H5DataSet,
  startFrame: number,
  windowSize: number,
  numChannels: number,
  totalSamples: number,
  layout: 'row-major' | 'column-major',
): Uint16Array {
  const endFrame = Math.min(startFrame + windowSize, totalSamples)

  if (layout === 'row-major') {
    // Data stored as [channels, samples] - each channel's data is contiguous
    // Read all channels for the window: [0:numChannels, startFrame:endFrame]
    return sig.slice([
      [0, numChannels],
      [startFrame, endFrame],
    ])
  } else {
    // Data stored as [samples, channels] - sample points are contiguous
    // Read the window then transpose: [startFrame:endFrame, 0:numChannels]
    const windowData = sig.slice([
      [startFrame, endFrame],
      [0, numChannels],
    ])

    // Need to transpose the data to match expected [channels, samples] format
    const actualFrames = endFrame - startFrame
    const transposed = new Uint16Array(numChannels * actualFrames)

    for (let channel = 0; channel < numChannels; channel++) {
      for (let frame = 0; frame < actualFrames; frame++) {
        // Source: [frame, channel], Dest: [channel, frame]
        transposed[channel * actualFrames + frame] = windowData[frame * numChannels + channel]
      }
    }

    return transposed
  }
}

// Number of values to retain for validation
const N_VALIDATION_VALUES = 10

interface ValidationReference {
  metadata: {
    h5_file: string
    model_path: string
    generated_by: string
    description: string
  }
  constants: {
    inference_scaling_numerator: number
    pre_median_frames: number
    sample_size: number
    num_output_locs: number
    input_scale: number
    buffer_front_sample: number
    buffer_end_sample: number
    n_validation_values: number
  }
  h5_file_parameters: {
    h5_file_path: string
    num_channels: number
    num_samples: number
    sampling_rate: number
    duration: number
    gain: number
    lsb: number
    hpf: number
    has_scaled_traces: boolean
    channel_gains: number[]
  }
  computed_values: {
    inference_scaling: number
    median_iqr: number
    iqrs_first_n: number[]
    iqrs_stats: {
      mean: number
      std: number
      min: number
      max: number
    }
  }
  sample_values: {
    raw_traces_before_scaling_channel_0: number[]
    raw_channel_0: number[]
    scaled_channel_0: number[]
    model_input_channel_0: number[]
    model_output_channel_0: number[]
  }
  model_processing: {
    window_start_frame: number
    window_end_frame: number
    num_channels: number
    window_data: number[][]
  }
}

/**
 * Validate H5 file parameters against reference
 */
function validateH5Parameters(
  parameters: H5FileParameters,
  reference: ValidationReference,
  tolerance: number = 1e-6,
): boolean {
  console.log('🔍 Validating H5 file parameters...')

  const ref = reference.h5_file_parameters
  let isValid = true

  // Check core parameters
  if (parameters.numChannels !== ref.num_channels) {
    console.error(`❌ Channel count mismatch: ${parameters.numChannels} vs ${ref.num_channels}`)
    isValid = false
  }

  if (Math.abs(parameters.samplingRate - ref.sampling_rate) > tolerance) {
    console.error(`❌ Sampling rate mismatch: ${parameters.samplingRate} vs ${ref.sampling_rate}`)
    isValid = false
  }

  if (Math.abs(parameters.gain - ref.gain) > tolerance) {
    console.error(`❌ Gain mismatch: ${parameters.gain} vs ${ref.gain}`)
    isValid = false
  }

  if (Math.abs(parameters.lsb - ref.lsb) > tolerance) {
    console.error(`❌ LSB mismatch: ${parameters.lsb} vs ${ref.lsb}`)
    isValid = false
  }

  if (
    parameters.hpf !== undefined &&
    ref.hpf !== undefined &&
    Math.abs(parameters.hpf - ref.hpf) > tolerance
  ) {
    console.error(`❌ HPF mismatch: ${parameters.hpf} vs ${ref.hpf}`)
    isValid = false
  }

  if (isValid) {
    console.log('✅ H5 file parameters validation passed')
  }

  return isValid
}

/**
 * Validate raw trace values (before scaling) against reference
 */
function validateRawTraceValues(
  rawFrames: Uint16Array,
  reference: ValidationReference,
  tolerance: number = 1e-9,
): boolean {
  console.log('🔍 Validating raw trace values (before scaling)...')

  let isValid = true

  // Check first N_VALIDATION_VALUES values of channel 0
  // Raw frames from h5wasm should match the raw traces before scaling from SpikeInterface
  for (let i = 0; i < N_VALIDATION_VALUES; i++) {
    const rawValue = rawFrames[i] // Direct uint16 value from h5 file
    const expectedValue = reference.sample_values.raw_traces_before_scaling_channel_0[i]

    if (Math.abs(rawValue - expectedValue) > tolerance) {
      console.error(
        `❌ Raw trace channel 0 value ${i} mismatch: ${rawValue} vs ${expectedValue}`,
      )
      isValid = false
    }
  }

  if (isValid) {
    console.log('✅ Raw trace values (before scaling) validation passed')
  }

  return isValid
}

/**
 * Validate raw H5 values against reference
 */
function validateRawValues(
  rawFrames: Uint16Array,
  parameters: H5FileParameters,
  reference: ValidationReference,
  tolerance: number = 1e-3,
): boolean {
  console.log('🔍 Validating raw H5 values...')

  // Convert raw values to physical values using same scaling as worker
  const scalingFactor = parameters.lsb * 1e6

  // Check first N_VALIDATION_VALUES values of channel 0
  const channel0Values: number[] = []

  for (let i = 0; i < N_VALIDATION_VALUES; i++) {
    channel0Values.push(rawFrames[i] * scalingFactor)
  }

  let isValid = true

  // Compare with reference
  for (let i = 0; i < N_VALIDATION_VALUES; i++) {
    if (Math.abs(channel0Values[i] - reference.sample_values.raw_channel_0[i]) > tolerance) {
      console.error(
        `❌ Channel 0 raw value ${i} mismatch: ${channel0Values[i]} vs ${reference.sample_values.raw_channel_0[i]}`,
      )
      isValid = false
    }
  }

  if (isValid) {
    console.log('✅ Raw H5 values validation passed')
  }

  return isValid
}

/**
 * Validate inference scaling calculation against reference
 */
function validateInferenceScaling(
  computedScaling: number,
  reference: ValidationReference,
  tolerance: number = 1e-6,
): boolean {
  console.log('🔍 Validating inference scaling...')

  const expectedScaling = reference.computed_values.inference_scaling
  const isValid = Math.abs(computedScaling - expectedScaling) <= tolerance

  if (!isValid) {
    console.error(`❌ Inference scaling mismatch: ${computedScaling} vs ${expectedScaling}`)
    console.error(`❌ Difference: ${Math.abs(computedScaling - expectedScaling)}`)
  } else {
    console.log(`✅ Inference scaling validation passed: ${computedScaling}`)
  }

  return isValid
}

/**
 * Validate model inputs and outputs against reference
 */
function validateModelData(
  modelInputs: Float16Array,
  modelOutputs: Float32Array,
  numChannels: number,
  reference: ValidationReference,
  tolerance: number = 1e-3,
): boolean {
  console.log('🔍 Validating model inputs and outputs...')

  let isValid = true

  // Check first N_VALIDATION_VALUES values of channel 0 model inputs
  for (let i = 0; i < N_VALIDATION_VALUES; i++) {
    const modelInputValue = modelInputs[i]

    if (Math.abs(modelInputValue - reference.sample_values.model_input_channel_0[i]) > tolerance) {
      console.error(
        `❌ Model input channel 0 value ${i} mismatch: ${modelInputValue} vs ${reference.sample_values.model_input_channel_0[i]}`,
      )
      isValid = false
    }
  }

  // Check first N_VALIDATION_VALUES values of channel 0 model outputs
  for (let i = 0; i < N_VALIDATION_VALUES; i++) {
    const modelOutputValue = modelOutputs[i]

    if (
      Math.abs(modelOutputValue - reference.sample_values.model_output_channel_0[i]) > tolerance
    ) {
      console.error(
        `❌ Model output channel 0 value ${i} mismatch: ${modelOutputValue} vs ${reference.sample_values.model_output_channel_0[i]}`,
      )
      isValid = false
    }
  }

  if (isValid) {
    console.log('✅ Model inputs and outputs validation passed')
  }

  return isValid
}

/**
 * Validate scaled values against reference
 */
function validateScaledValues(
  scaledTraces: Float16Array,
  reference: ValidationReference,
  tolerance: number = 1e-7,
): boolean {
  console.log('🔍 Validating scaled values...')

  let isValid = true

  // Check first N_VALIDATION_VALUES values of channel 0
  for (let i = 0; i < N_VALIDATION_VALUES; i++) {
    const scaledValue = scaledTraces[i]

    if (Math.abs(scaledValue - reference.sample_values.scaled_channel_0[i]) > tolerance) {
      console.error(
        `❌ Scaled channel 0 value ${i} mismatch: ${scaledValue} vs ${reference.sample_values.scaled_channel_0[i]}`,
      )
      isValid = false
    }
  }

  if (isValid) {
    console.log('✅ Scaled values validation passed')
  }

  return isValid
}

/**
 * Run comprehensive validation against reference data
 */
async function runValidationWithData(
  modelsURL: string,
  parameters: H5FileParameters,
  inferenceScaling: number,
  rawFrames: Uint16Array,
  scaledTraces: Float16Array,
  modelInputs: Float16Array,
  modelOutputs: Float32Array,
): Promise<boolean> {
  console.log('🔍 Starting comprehensive validation...')

  try {
    // Load reference data
    const response = await fetch(`${modelsURL}/test_maxwell_raw.validation.json`)
    if (!response.ok) {
      throw new Error(`Failed to fetch reference data: ${response.statusText}`)
    }
    const reference = (await response.json()) as ValidationReference
    console.log('📋 Loaded validation reference data')

    let allValid = true

    // 1. Validate H5 parameters
    if (!validateH5Parameters(parameters, reference)) {
      allValid = false
    }

    // 2. Validate raw trace values (before scaling)
    if (!validateRawTraceValues(rawFrames, reference)) {
      allValid = false
    }

    // 3. Validate raw values (first N values from channel 0)
    if (!validateRawValues(rawFrames, parameters, reference)) {
      allValid = false
    }

    // 4. Validate scaled values (first N values from channel 0)
    if (!validateScaledValues(scaledTraces, reference)) {
      allValid = false
    }

    // 5. Validate inference scaling
    if (!validateInferenceScaling(inferenceScaling, reference)) {
      allValid = false
    }

    // 6. Validate model inputs and outputs (first N values from channel 0)
    if (!validateModelData(modelInputs, modelOutputs, parameters.numChannels, reference)) {
      allValid = false
    }

    if (allValid) {
      console.log('🎉 All validations passed!')
    } else {
      console.log('❌ Some validations failed!')
    }

    return allValid
  } catch (error) {
    console.warn('⚠️ Validation could not be completed:', error)
    return false
  }
}

/**
 * Run the detection model on the h5 file using streaming
 */
async function* runDetectionModel(
  h5File: File,
  parameters: H5FileParameters,
  inferenceScaling: number,
  detectSession: InferenceSession,
): AsyncGenerator<
  {
    output: Float32Array
    duration: number
    validationData?: { modelInputs: Float16Array; modelOutputs: Float32Array }
  },
  void,
  unknown
> {
  const sampleSize = 200
  const inputScale = 0.15887516

  const numChannels = parameters.numChannels
  const totalSamples = parameters.numSamples
  const samplingRate = parameters.samplingRate

  // Log storage optimization info
  console.log(`💾 Using ${parameters.storageLayout} optimized reading`)
  if (parameters.isChunked && parameters.chunkSize) {
    console.log(`📦 File is chunked: [${parameters.chunkSize.join(', ')}]`)
  }

  // Performance tracking metrics
  let totalSamplesRead = 0 // Total samples read from file (frames × channels)
  let totalSamplesProcessed = 0 // Total samples processed through model (frames × channels)
  let totalReadTime = 0 // Total time spent reading from file (ms)
  let totalProcessTime = 0 // Total time spent processing/inference (ms)

  let startTime = performance.now()
  let lastReportTime = startTime
  const reportInterval = 1000 // Report every 1 second

  let isFirstWindow = true

  // Initialize h5wasm and handle mounting here
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

    // Process windows
    for (let startFrame = 0; startFrame <= totalSamples - sampleSize; startFrame += sampleSize) {
      const windowStartTime = performance.now()

      // === FILE READING PHASE ===
      const readStartTime = performance.now()

      // Read this window from the h5 file using optimized pattern
      const endFrame = Math.min(startFrame + sampleSize, totalSamples)
      const actualFrames = endFrame - startFrame

      if (actualFrames <= 0) continue

      // Use optimized reading based on storage layout
      const frames = readWindowOptimized(
        sig,
        startFrame,
        sampleSize,
        numChannels,
        totalSamples,
        parameters.storageLayout,
      )

      const readEndTime = performance.now()
      const readDuration = readEndTime - readStartTime
      totalReadTime += readDuration
      totalSamplesRead += actualFrames * numChannels

      // === PROCESSING PHASE ===
      const processStartTime = performance.now()

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

      const processEndTime = performance.now()
      const processDuration = processEndTime - processStartTime
      totalProcessTime += processDuration
      totalSamplesProcessed += actualFrames * numChannels

      const windowEndTime = performance.now()
      const windowDuration = windowEndTime - windowStartTime

      // Check if it's time to report realtime capacity
      const currentTime = performance.now()
      if (currentTime - lastReportTime >= reportInterval) {
        const elapsedSeconds = (currentTime - startTime) / 1000

        // Calculate throughput rates (samples per second)
        const readSamplesPerSecond = totalSamplesRead / (totalReadTime / 1000)
        const processSamplesPerSecond = totalSamplesProcessed / (totalProcessTime / 1000)
        const overallSamplesPerSecond = totalSamplesProcessed / elapsedSeconds

        // Calculate realtime channel capacities
        const readChannelCapacity = readSamplesPerSecond / samplingRate
        const processChannelCapacity = processSamplesPerSecond / samplingRate
        const overallChannelCapacity = overallSamplesPerSecond / samplingRate

        // Calculate time distribution percentages
        const totalActiveTime = totalReadTime + totalProcessTime
        const readPercentage = totalActiveTime > 0 ? (totalReadTime / totalActiveTime) * 100 : 0
        const processPercentage =
          totalActiveTime > 0 ? (totalProcessTime / totalActiveTime) * 100 : 0

        // Send comprehensive performance update to main thread
        self.postMessage({
          type: 'realtimeCapacity',
          // Overall metrics
          capacity: overallChannelCapacity,
          samplesPerSecond: overallSamplesPerSecond,
          // Breakdown by phase
          readChannelCapacity: readChannelCapacity,
          processChannelCapacity: processChannelCapacity,
          readSamplesPerSecond: readSamplesPerSecond,
          processSamplesPerSecond: processSamplesPerSecond,
          // Time distribution
          readPercentage: readPercentage,
          processPercentage: processPercentage,
          totalReadTime: totalReadTime,
          totalProcessTime: totalProcessTime,
          // Context
          totalChannels: numChannels,
          samplingRate: samplingRate,
          elapsedSeconds: elapsedSeconds,
        })

        lastReportTime = currentTime
      }

      // Send progress update to the main thread
      self.postMessage({
        type: 'processingProgress',
        message: 'Detecting spikes...',
        countFinished: startFrame + sampleSize,
        totalToProcess: totalSamples,
      })

      // Collect validation data only for the first window
      let validationData: { modelInputs: Float16Array; modelOutputs: Float32Array } | undefined

      if (isFirstWindow) {
        // Extract first N_VALIDATION_VALUES from channel 0 for validation
        const modelInputs = new Float16Array(N_VALIDATION_VALUES)
        const modelOutputs = new Float32Array(N_VALIDATION_VALUES)

        for (let i = 0; i < N_VALIDATION_VALUES; i++) {
          modelInputs[i] = processedData[i] // Channel 0 data
          modelOutputs[i] = (results.output.data as Float32Array)[i] // Channel 0 output
        }

        validationData = { modelInputs, modelOutputs }
        isFirstWindow = false
      }

      yield {
        output: results.output.data as Float32Array,
        duration: windowDuration,
        validationData,
      }
    }

    // Clean up
    h5.close()
    FS.unmount('/work')

    self.postMessage({
      type: 'processingProgress',
      message: 'Processing completed',
      countFinished: parameters.numSamples,
      totalToProcess: parameters.numSamples,
    })
  } catch (error) {
    console.error('❌ Error in detection model with h5 streaming:', error)

    // Clean up on error
    try {
      FS.unmount('/work')
    } catch (e) {
      // Ignore unmount errors
    }

    throw error
  }
}

self.addEventListener('message', async function (event: MessageEvent<PredictionMessage>) {
  if (event.data.type === 'open' && event.data.file) {
    console.log('Received open message')
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
  } else if (event.data.type === 'run' && event.data.file) {
    console.log('Received runWithH5 message')
    console.log('Models URL:', event.data.modelsURL)
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
      const {
        inferenceScaling: inferenceScalingValue,
        rawFrames,
        scaledTraces,
      } = await calculateInferenceScalingFromH5WithValidation(event.data.file, parameters)
      console.log(`✅ Inference scaling computed: ${inferenceScalingValue}`)

      // Run detection model with h5 streaming
      console.log('🔄 Running detection model with h5 streaming...')
      const detectionResults: Float32Array[] = []
      let firstWindowValidationData:
        | { modelInputs: Float16Array; modelOutputs: Float32Array }
        | undefined

      for await (const { output, validationData } of runDetectionModel(
        event.data.file,
        parameters,
        inferenceScalingValue,
        detectSession,
      )) {
        detectionResults.push(output)

        // Capture validation data from first window
        if (validationData && !firstWindowValidationData) {
          firstWindowValidationData = validationData
        }
      }

      console.log(`✅ Detection completed. Generated ${detectionResults.length} windows`)

      // Run validation against reference data
      console.log('🔍 Running validation against reference data...')
      const validationPassed =
        event.data.modelsURL && firstWindowValidationData
          ? await runValidationWithData(
              event.data.modelsURL,
              parameters,
              inferenceScalingValue,
              rawFrames,
              scaledTraces,
              firstWindowValidationData.modelInputs,
              firstWindowValidationData.modelOutputs,
            )
          : false

      // Send results back to main thread
      self.postMessage({
        type: 'result',
        result: {
          inferenceScaling: inferenceScalingValue,
          detectionOutputs: detectionResults,
          parameters,
          executionProvider: executionProviders[0],
          totalWindows: detectionResults.length,
          validationPassed,
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
