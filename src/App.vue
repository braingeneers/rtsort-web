<template>
  <v-app theme="dark">
    <v-main>
      <v-container>
        <h1>RT-Sort Web</h1>
        <v-card class="mb-4">
          <v-card-title>Test</v-card-title>
          <v-card-text>
            <v-btn @click="start" color="primary" class="mr-2">
              {{ running ? 'Running...' : 'Run' }}
            </v-btn>
          </v-card-text>
        </v-card>
      </v-container>
    </v-main>
  </v-app>
</template>

<script lang="ts" setup>
import { onMounted, ref } from 'vue'
import SIMSWorker from './worker.ts?worker'

const worker = ref<Worker | null>(null)
const running = ref(false)

/**
 * Load expected model outputs from model_outputs.bin for comparison
 */
async function loadExpectedModelOutputs(): Promise<Float32Array> {
  const response = await fetch(`${window.location.href}models/model_outputs.bin`)
  if (!response.ok) {
    throw new Error(`Failed to fetch model_outputs.bin: ${response.statusText}`)
  }
  const arrayBuffer = await response.arrayBuffer()

  // Convert from float16 (stored as uint16) to float32 for comparison
  const uint16Data = new Uint16Array(arrayBuffer)
  const float32Data = new Float32Array(uint16Data.length)

  // Convert each float16 value to float32
  for (let i = 0; i < uint16Data.length; i++) {
    float32Data[i] = float16ToFloat32(uint16Data[i])
  }

  return float32Data
}

/**
 * Convert float16 stored as uint16 to float32
 */
function float16ToFloat32(uint16Value: number): number {
  const sign = (uint16Value & 0x8000) >> 15
  const exponent = (uint16Value & 0x7c00) >> 10
  const mantissa = uint16Value & 0x03ff

  let result: number

  if (exponent === 0) {
    if (mantissa === 0) {
      result = 0
    } else {
      result = Math.pow(2, -14) * (mantissa / 1024)
    }
  } else if (exponent === 0x1f) {
    if (mantissa === 0) {
      result = Infinity
    } else {
      result = NaN
    }
  } else {
    result = Math.pow(2, exponent - 15) * (1 + mantissa / 1024)
  }

  return sign ? -result : result
}

/**
 * Verify that worker outputs match expected model outputs for first 5 windows
 */
async function verifyModelOutputs(workerResults: Float32Array[], expectedData: Float32Array) {
  const numChannels = 942
  const framesPerWindow = 120
  const windowsToCheck = 5

  console.log('\n🔍 Verifying model outputs...')
  console.log(`Expected data shape: [${numChannels}, ${expectedData.length / numChannels}]`)
  console.log(`Worker results: ${workerResults.length} windows`)
  console.log(
    `Checking first ${windowsToCheck} windows (${windowsToCheck * framesPerWindow} frames)`
  )

  if (workerResults.length < windowsToCheck) {
    console.log(`❌ Not enough worker results: ${workerResults.length} < ${windowsToCheck}`)
    return false
  }

  let totalMatches = 0
  let totalComparisons = 0
  const tolerance = 1e-2 // 2 decimal places

  for (let window = 0; window < windowsToCheck; window++) {
    const workerOutput = workerResults[window]
    let windowMatches = 0
    let windowComparisons = 0

    for (let channel = 0; channel < numChannels; channel++) {
      for (let frame = 0; frame < framesPerWindow; frame++) {
        const workerValue = workerOutput[channel * framesPerWindow + frame]
        const expectedIndex =
          channel * (expectedData.length / numChannels) + (window * framesPerWindow + frame)
        const expectedValue = expectedData[expectedIndex]

        const matches = Math.abs(workerValue - expectedValue) < tolerance
        if (matches) windowMatches++
        windowComparisons++
        totalComparisons++
        if (matches) totalMatches++
        if (!matches) {
          console.warn(
            `Mismatch at window ${window}, channel ${channel}, frame ${frame}: worker=${workerValue.toFixed(
              6
            )}, expected=${expectedValue.toFixed(6)}`
          )
        }

        // Log first few comparisons for debugging
        if (window === 0 && channel === 0 && frame < 5) {
          console.log(
            `  Frame ${frame}: worker=${workerValue.toFixed(6)}, expected=${expectedValue.toFixed(
              6
            )}, match=${matches}`
          )
        }
      }
    }

    const windowMatchPercentage = (windowMatches / windowComparisons) * 100
    console.log(
      `  Window ${window}: ${windowMatches}/${windowComparisons} matches (${windowMatchPercentage.toFixed(
        1
      )}%)`
    )
  }

  const totalMatchPercentage = (totalMatches / totalComparisons) * 100
  const success = totalMatchPercentage > 95 // Require 95% of values to match within tolerance

  console.log(
    `\n📊 Overall: ${totalMatches}/${totalComparisons} matches (${totalMatchPercentage.toFixed(
      1
    )}%)`
  )
  console.log(`${success ? '✅ MODEL OUTPUTS MATCH' : '❌ MODEL OUTPUTS DO NOT MATCH'}`)

  return success
}

function initializeWorkers() {
  console.log('Initializing workers...')
  running.value = true
  worker.value = new SIMSWorker()
  worker.value.onmessage = async (event) => {
    if (event.data.type === 'result') {
      running.value = false
      const result = event.data.result
      console.log('Received result:', result)

      // Verify inference scaling
      console.log(`Expected inference scaling: 0.3761194050`)
      const inferenceScalingMatch = Math.abs(result.inferenceScaling - 0.376119405) < 1e-6
      console.log(`Inference scaling match: ${inferenceScalingMatch ? '✅ PERFECT' : '❌ FAILED'}`)

      // Verify model outputs
      try {
        const expectedModelOutputs = await loadExpectedModelOutputs()
        await verifyModelOutputs(result.detectionOutputs, expectedModelOutputs)
      } catch (error) {
        console.error('❌ Failed to verify model outputs:', error)
      }
    }
  }
}

function start() {
  if (!worker.value) return

  worker.value.postMessage({
    type: 'start',
    modelsURL: `${window.location.href}models`,
  })
  running.value = true
}

onMounted(() => {
  initializeWorkers()
  start()
})
</script>
