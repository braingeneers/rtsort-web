<template>
  <v-app theme="dark">
    <v-main>
      <v-container>
        <h1>RT-Sort Web</h1>
        <v-file-input
          v-model="selectedFile"
          data-cy="file-input"
          accept=".h5"
          label="Select File"
          variant="outlined"
          prepend-icon=""
          show-size
          @update:model-value="handleFileSelected"
        ></v-file-input>

        <!-- File Parameters Display -->
        <v-card v-if="fileParameters" variant="outlined" class="mt-4">
          <v-card-title class="text-h6">File Parameters</v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="6" sm="4">
                <div class="text-caption">Channels</div>
                <div class="text-body-1">{{ fileParameters.numChannels }}</div>
              </v-col>
              <v-col cols="6" sm="4">
                <div class="text-caption">Frames</div>
                <div class="text-body-1">{{ fileParameters.numSamples }}</div>
              </v-col>
              <v-col cols="6" sm="4">
                <div class="text-caption">Duration</div>
                <div class="text-body-1">{{ fileParameters.duration.toFixed(2) }}s</div>
              </v-col>
              <v-col cols="6" sm="4">
                <div class="text-caption">Sampling Rate</div>
                <div class="text-body-1">{{ fileParameters.samplingRate }} Hz</div>
              </v-col>
              <v-col cols="6" sm="4">
                <div class="text-caption">LSB</div>
                <div class="text-body-1">{{ fileParameters.lsb }}</div>
              </v-col>
              <v-col cols="6" sm="4">
                <div class="text-caption">Gain</div>
                <div class="text-body-1">{{ fileParameters.gain }}</div>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>

        <v-switch
          v-model="useGPU"
          label="Use GPU (WebGPU/WebGL)"
          color="primary"
          hide-details
          :disabled="running"
          class="mt-4"
        ></v-switch>

        <div class="mt-4">
          <v-btn
            v-if="!running"
            @click="handleRun"
            color="primary"
            class="mr-2"
            :disabled="parsingFile"
            data-cy="run-button"
          >
            Run
          </v-btn>
          <v-btn v-else @click="handleStop" color="error" class="mr-2" data-cy="stop-button">
            Stop
          </v-btn>
          <v-chip :color="useGPU ? 'green' : 'blue'" size="small">
            {{ useGPU ? 'GPU' : 'CPU' }}
          </v-chip>
        </div>
        <div class="text-caption mt-1">{{ currentStatus }}</div>
        <div v-if="running" class="pa-2">
          <v-progress-linear
            :model-value="processingProgress"
            color="primary"
            height="4"
          ></v-progress-linear>
        </div>

        <!-- Realtime Capacity Display -->
        <v-card v-if="realtimeCapacity !== null" variant="outlined" class="mb-4">
          <v-card-title class="text-h6">
            <v-icon class="mr-2">mdi-speedometer</v-icon>
            Realtime Processing Capacity
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="6" sm="3">
                <div class="text-caption">Channels in Realtime</div>
                <div
                  class="text-h4"
                  :class="
                    realtimeCapacity >= fileParameters?.numChannels ? 'text-green' : 'text-orange'
                  "
                >
                  {{ realtimeCapacity.toFixed(1) }}
                </div>
                <div class="text-caption">of {{ fileParameters?.numChannels || 0 }} total</div>
              </v-col>
              <v-col cols="6" sm="3">
                <div class="text-caption">Processing Speed</div>
                <div class="text-body-1">{{ framesPerSecond?.toFixed(0) || 0 }} frames/sec</div>
                <div class="text-caption">{{ samplesPerSecond?.toFixed(0) || 0 }} samples/sec</div>
              </v-col>
              <v-col cols="6" sm="3">
                <div class="text-caption">Required Speed</div>
                <div class="text-body-1">{{ fileParameters?.samplingRate || 0 }} frames/sec</div>
                <div class="text-caption">
                  {{
                    (
                      (fileParameters?.samplingRate || 0) * (fileParameters?.numChannels || 0)
                    ).toFixed(0)
                  }}
                  samples/sec
                </div>
              </v-col>
              <v-col cols="6" sm="3">
                <div class="text-caption">Efficiency</div>
                <div class="text-body-1" :class="efficiency >= 100 ? 'text-green' : 'text-orange'">
                  {{ efficiency.toFixed(1) }}%
                </div>
              </v-col>
            </v-row>
            <v-progress-linear
              :model-value="Math.min(efficiency, 100)"
              :color="efficiency >= 100 ? 'green' : 'orange'"
              height="8"
              class="mt-2"
            ></v-progress-linear>
          </v-card-text>
        </v-card>

        <!-- Error Section -->
        <v-alert v-if="errorMessage" type="error" class="mb-4">
          {{ errorMessage }}
        </v-alert>
      </v-container>
    </v-main>
  </v-app>
</template>

<script lang="ts" setup>
import { onMounted, ref, computed } from 'vue'
import RTSortWorker from './worker.ts?worker'

const worker = ref<Worker | null>(null)
const parsingFile = ref(false)
const running = ref(false)
const useGPU = ref(false)
const currentStatus = ref('')
const processingProgress = ref(0)
const benchmarks = ref<{
  avgTime: number
  minTime: number
  maxTime: number
  provider: string
  totalWindows: number
} | null>(null)

// New realtime capacity tracking
const realtimeCapacity = ref<number | null>(null)
const framesPerSecond = ref<number | null>(null)
const samplesPerSecond = ref<number | null>(null)

// Computed efficiency percentage
const efficiency = computed(() => {
  if (!realtimeCapacity.value || !fileParameters.value) return 0
  return (realtimeCapacity.value / fileParameters.value.numChannels) * 100
})

const errorMessage = ref<string | null>(null)

const selectedFile = ref<File | null>(null)
const fileParameters = ref<{
  numChannels: number
  numSamples: number
  samplingRate: number
  gain: number
  lsb: number
  duration: number
} | null>(null)

function handleFileSelected(files: File | File[]) {
  parsingFile.value = true
  const file = Array.isArray(files) ? files[0] : files
  if (file) {
    selectedFile.value = file
    fileParameters.value = null // Reset parameters
    // Send message to worker to extract h5 file parameters
    if (file.name.endsWith('.h5')) {
      worker.value?.postMessage({
        type: 'openFile',
        file: file,
      })
    }
  } else {
    selectedFile.value = null
    fileParameters.value = null
    parsingFile.value = false
  }
}

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
    `Checking first ${windowsToCheck} windows (${windowsToCheck * framesPerWindow} frames)`,
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
        // if (!matches) {
        //   console.warn(
        //     `Mismatch at window ${window}, channel ${channel}, frame ${frame}: worker=${workerValue.toFixed(
        //       6
        //     )}, expected=${expectedValue.toFixed(6)}`
        //   )
        // }

        // Log first few comparisons for debugging
        if (window === 0 && channel === 0 && frame < 5) {
          console.log(
            `  Frame ${frame}: worker=${workerValue.toFixed(6)}, expected=${expectedValue.toFixed(
              6,
            )}, match=${matches}`,
          )
        }
      }
    }

    const windowMatchPercentage = (windowMatches / windowComparisons) * 100
    console.log(
      `  Window ${window}: ${windowMatches}/${windowComparisons} matches (${windowMatchPercentage.toFixed(
        1,
      )}%)`,
    )
  }

  const totalMatchPercentage = (totalMatches / totalComparisons) * 100
  const success = totalMatchPercentage > 95 // Require 95% of values to match within tolerance

  console.log(
    `\n📊 Overall: ${totalMatches}/${totalComparisons} matches (${totalMatchPercentage.toFixed(
      1,
    )}%)`,
  )
  console.log(`${success ? '✅ MODEL OUTPUTS MATCH' : '❌ MODEL OUTPUTS DO NOT MATCH'}`)

  return success
}

function initializeWorkers() {
  console.log('Initializing workers...')
  worker.value = new RTSortWorker()
  worker.value.onmessage = async (event) => {
    if (event.data.type === 'processingProgress') {
      // Update progress based on countFinished and totalToProcess
      const { countFinished, totalToProcess } = event.data
      processingProgress.value = (countFinished / totalToProcess) * 100
      currentStatus.value = `Processing: ${countFinished} of ${totalToProcess} complete (${Math.round(processingProgress.value)}%)`
    } else if (event.data.type === 'realtimeCapacity') {
      // Handle realtime capacity updates
      realtimeCapacity.value = event.data.capacity
      framesPerSecond.value = event.data.framesPerSecond
      samplesPerSecond.value = event.data.samplesPerSecond
      console.log(
        `📈 Realtime capacity: ${event.data.capacity.toFixed(1)} channels (${event.data.framesPerSecond.toFixed(0)} frames/sec, ${event.data.samplesPerSecond.toFixed(0)} samples/sec)`,
      )
    } else if (event.data.type === 'fileParameters') {
      // Handle file parameters response
      fileParameters.value = event.data.parameters
      parsingFile.value = false
      errorMessage.value = null
      currentStatus.value = 'File parameters extracted'
    } else if (event.data.type === 'stopped') {
      // Handle stopped message from worker
      running.value = false
      errorMessage.value = null
      currentStatus.value = 'Processing stopped by user'
      processingProgress.value = 0
      realtimeCapacity.value = null
      framesPerSecond.value = null
      samplesPerSecond.value = null
      console.log('Processing stopped by user')
    } else if (event.data.type === 'result') {
      running.value = false
      errorMessage.value = null
      const result = event.data.result
      console.log('Received result:', result)

      // Store minimal result info
      benchmarks.value = {
        avgTime: 0,
        minTime: 0,
        maxTime: 0,
        provider: result.executionProvider,
        totalWindows: result.totalWindows,
      }

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
    } else if (event.data.type === 'error') {
      running.value = false
      errorMessage.value = event.data.error
      realtimeCapacity.value = null
      framesPerSecond.value = null
      samplesPerSecond.value = null
      console.error('Worker error:', event.data.error)
    }
  }

  worker.value.onerror = (error) => {
    running.value = false
    errorMessage.value = `Worker error: ${error.message}`
    console.error('Worker error:', error)
  }
}

function handleStop() {
  console.log('🛑 Stopping worker...')

  // Terminate the current worker
  if (worker.value) {
    worker.value.terminate()
    worker.value = null
  }

  // Reset state
  running.value = false
  processingProgress.value = 0
  currentStatus.value = 'Processing stopped'
  errorMessage.value = null
  realtimeCapacity.value = null
  framesPerSecond.value = null
  samplesPerSecond.value = null

  // Re-initialize worker for future use
  initializeWorkers()
}

function handleRun() {
  if (!selectedFile.value) {
    errorMessage.value = 'Please select a file to start.'
    return
  }

  if (selectedFile.value.name.endsWith('.h5')) {
    if (!fileParameters.value) {
      errorMessage.value = 'File parameters not extracted yet. Please wait or reselect the file.'
      return
    }
    // Use h5 workflow
    running.value = true
    errorMessage.value = null
    benchmarks.value = null
    currentStatus.value = 'Starting spike detector with h5 file...'
    processingProgress.value = 0
    worker.value?.postMessage({
      type: 'runWithH5',
      file: selectedFile.value,
      modelsURL: `${window.location.href}models`,
      useGPU: useGPU.value,
    })
  } else {
    // Use legacy workflow
    running.value = true
    errorMessage.value = null
    benchmarks.value = null
    currentStatus.value = 'Starting spike detector...'
    processingProgress.value = 0
    worker.value?.postMessage({
      type: 'start',
      modelsURL: `${window.location.href}models`,
      useGPU: useGPU.value,
    })
  }
}

// Fetch a sample file on application load
async function fetchSampleFile() {
  try {
    const sampleFileName = 'sample_maxwell_raw.h5'
    currentStatus.value = 'Loading sample recording...'

    const response = await fetch(sampleFileName)
    const blob = await response.blob()
    const file = new File([blob], sampleFileName, { type: blob.type })

    selectedFile.value = file
    currentStatus.value = 'Sample file loaded'
    console.log('Sample File:', file)

    // Extract file parameters if it's an h5 file
    if (file.name.endsWith('.h5') && worker.value) {
      worker.value.postMessage({
        type: 'openFile',
        file: file,
      })
    }
  } catch (error) {
    console.error('Error loading sample file:', error)
    currentStatus.value = 'Error loading sample file'
  }
}

onMounted(async () => {
  initializeWorkers()
  await fetchSampleFile()
  // errorMessage.value = 'Please select a file to start.'
})
</script>

<style scoped>
/* Remove any custom styles that might be hiding the file input */
</style>
