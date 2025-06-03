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
 * @param inferenceScalingSession - The ONNX inference session for scaling
 * @param modelsURL - Base URL for the model files
 * @returns Promise<number | object> - The computed inference scaling value or performance metrics
 */
async function computeInferenceScaling(
  inferenceScalingSession: InferenceSession,
  modelsURL: string
): Promise<number | any> {
  try {
    console.log('🔄 Fetching scaled_traces.bin...')
    const response = await fetch(`${modelsURL}/../scaled_traces.bin`)
    if (!response.ok) {
      throw new Error(`Failed to fetch scaled_traces.bin: ${response.statusText}`)
    }

    console.log('🔄 Parsing binary data...')
    const arrayBuffer = await response.arrayBuffer()

    // Based on the metadata: shape [942, 2000], dtype float16
    const totalElements = 942 * 2000
    const expectedBytes = totalElements * 2 // 2 bytes per float16

    console.log(`📊 File size: ${arrayBuffer.byteLength} bytes (expected: ${expectedBytes} bytes)`)

    if (arrayBuffer.byteLength !== expectedBytes) {
      throw new Error(
        `Unexpected file size. Expected ${expectedBytes} bytes, got ${arrayBuffer.byteLength}`
      )
    }

    console.log('🔄 Creating tensor from full ArrayBuffer...')
    // Parse as float16 data (942 channels, 2000 samples each)
    const numChannels = 942
    const totalSamples = 2000
    const windowSize = 1000 // First 1000 samples

    const windowDataViews = new Uint16Array(numChannels * windowSize)

    // Create views for each channel and copy in batches
    for (let channel = 0; channel < numChannels; channel++) {
      const srcStart = channel * totalSamples
      const dstStart = channel * windowSize

      // Create a view of the source data for this channel's window
      const sourceView = new Uint16Array(
        arrayBuffer,
        srcStart * 2, // byte offset (2 bytes per Uint16)
        windowSize // length in elements
      )

      // Copy using set() which should be optimized for typed arrays
      windowDataViews.set(sourceView, dstStart)
    }

    const windowData = windowDataViews

    console.log('🔄 Creating ONNX tensors from extracted window...')
    // Create input tensors for ONNX (data is already Float16)
    const windowTensor = new Tensor('float16', windowData, [numChannels, windowSize])
    const inferenceScalingNumerator = new Tensor('float32', new Float32Array([12.6]), [])

    console.log('🔄 Running ONNX inference...')
    // Run inference
    const results = await inferenceScalingSession.run({
      window: windowTensor,
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

    // Compute the inference scaling value
    try {
      console.log('Starting inference scaling computation...')
      const inferenceScalingValue = (await computeInferenceScaling(
        inferenceScalingSession,
        event.data.modelsURL
      )) as number
      console.log(`✅ Inference scaling computed successfully: ${inferenceScalingValue}`)

      // Send results back to main thread
      self.postMessage({
        type: 'result',
        result: inferenceScalingValue,
      })
    } catch (error) {
      console.error('❌ Failed to compute inference scaling:', error)
    }
  }
})
