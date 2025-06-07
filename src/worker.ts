/**
 * Browser web worker for running rt-sort models via onnx
 */

import { InferenceSession, Tensor, env } from 'onnxruntime-web'

interface PredictionMessage {
  type: 'start'
  modelsURL: string
}

/**
 * Compute inference scaling using the first 1000 samples from scaled_traces.bin
 * @param scaledTraces - The Float16Array containing scaled traces
 * @param inferenceScalingSession - The ONNX inference session for scaling
 * @returns Promise<number | object> - The computed inference scaling value or performance metrics
 */
async function computeInferenceScaling(
  scaledTraces: Float16Array,
  inferenceScalingSession: InferenceSession
): Promise<number | any> {
  try {
    console.log('🔄 Creating ONNX tensors from extracted window...')
    // Create input tensors for ONNX (data is already Float16)
    // Assuming numChannels and windowSize are known or passed if they vary.
    // For this refactor, let's assume they are constants or can be derived.
    // Based on previous logic: numChannels = 942, windowSize = 1000
    const numChannels = 942 // This might need to be dynamic if it changes
    const windowSize = 1000 // This might need to be dynamic if it changes
    const inferenceScalingNumerator = new Tensor('float32', new Float32Array([12.6]), [])

    const scaledTracesTensor = new Tensor(
      'float16',
      scaledTraces.slice(0, numChannels * windowSize),
      [numChannels, windowSize]
    )

    console.log('Scaled traces tensor created with shape:', scaledTracesTensor.dims)
    console.log(
      'Inference scaling numerator tensor created with shape:',
      inferenceScalingNumerator.dims
    )

    console.log('🔄 Running ONNX inference...')
    // Run inference
    const results = await inferenceScalingSession.run({
      window: scaledTracesTensor,
      inference_scaling_numerator: inferenceScalingNumerator,
    })

    const inferenceScaling = results.inference_scaling.data[0] as number
    console.log(`📈 Raw inference scaling result: ${inferenceScaling}`)

    return inferenceScaling
  } catch (error) {
    console.error('💥 Error in computeInferenceScaling:', error)
    throw error
  }
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

  // REMIND
  inferenceScaling = 0.3761194029850746

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
    const inputTensor = new Tensor('float16', processedData, [numChannels, 1, sampleSize])

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

    const inferenceScalingSession = await InferenceSession.create(
      `${event.data.modelsURL}/inference_scaling.onnx`,
      options
    )
    console.log('Inference scaling output names', inferenceScalingSession.outputNames)

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
      const inferenceScalingValue = (await computeInferenceScaling(
        scaledTraces,
        inferenceScalingSession
      )) as number
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
