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
          class="mt-4"
          color="white"
          base-color="white"
        ></v-file-input>

        <!-- File Parameters Display -->
        <v-card v-if="fileParameters" variant="outlined" class="mt-4">
          <v-card-text class="pb-2">
            <v-row dense>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">Channels</div>
                <div class="text-body-1">{{ fileParameters.numChannels }}</div>
              </v-col>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">Frames</div>
                <div class="text-body-1">{{ fileParameters.numSamples }}</div>
              </v-col>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">Duration</div>
                <div class="text-body-1">{{ fileParameters.duration.toFixed(2) }}s</div>
              </v-col>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">Sampling Rate</div>
                <div class="text-body-1">{{ fileParameters.samplingRate }} Hz</div>
              </v-col>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">LSB</div>
                <div class="text-body-1">{{ fileParameters.lsb }}</div>
              </v-col>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">Gain</div>
                <div class="text-body-1">{{ fileParameters.gain }}</div>
              </v-col>
              <v-col cols="6" sm="4" class="py-1">
                <div class="text-caption">Storage Layout</div>
                <div class="text-body-1">
                  {{ fileParameters.storageLayout }}
                  <v-chip
                    size="x-small"
                    :color="fileParameters.isChunked ? 'blue' : 'grey'"
                    class="ml-1"
                  >
                    {{ fileParameters.isChunked ? 'chunked' : 'contiguous' }}
                  </v-chip>
                  <v-chip
                    v-if="fileParameters.fileFormat"
                    size="x-small"
                    :color="
                      fileParameters.fileFormat === 'maxwell'
                        ? 'green'
                        : fileParameters.fileFormat === 'nwb'
                          ? 'purple'
                          : 'orange'
                    "
                    class="ml-1"
                  >
                    {{ fileParameters.fileFormat }}
                  </v-chip>
                </div>
                <div v-if="fileParameters.chunkSize" class="text-caption text-medium-emphasis">
                  Chunks: [{{ fileParameters.chunkSize.join(' × ') }}]
                </div>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>

        <!-- GPU/CPU Selection Radio Buttons -->
        <v-card variant="outlined" class="mt-4">
          <v-card-title class="text-subtitle-1 pb-2">Processing Mode</v-card-title>
          <v-card-text class="pt-0">
            <v-radio-group v-model="processingMode" inline class="ma-1 mb-0">
              <v-radio label="CPU" value="cpu" color="primary" :disabled="running"></v-radio>
              <v-radio
                label="GPU"
                value="gpu"
                color="primary"
                :disabled="running || !gpuAvailable"
              ></v-radio>
            </v-radio-group>
            <div class="text-caption text-medium-emphasis">
              <v-icon size="small" :color="gpuAvailable ? 'success' : 'warning'" class="mr-1">
                {{ gpuAvailable ? 'mdi-check-circle' : 'mdi-alert-circle' }}
              </v-icon>
              {{ gpuStatusText }}
            </div>
          </v-card-text>
        </v-card>

        <div class="mt-2">
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
          <v-chip :color="processingMode === 'gpu' ? 'green' : 'blue'" size="small">
            {{ processingMode === 'gpu' ? 'GPU' : 'CPU' }}
          </v-chip>
          <div data-cy="status" class="text-caption mt-4">{{ currentStatus }}</div>
        </div>
        <div v-if="running" class="pa-2">
          <v-progress-linear
            :model-value="processingProgress"
            color="primary"
            height="4"
          ></v-progress-linear>
        </div>

        <!-- Realtime Capacity Display -->
        <v-card v-if="realtimeCapacity !== null" variant="outlined" class="mt-4">
          <v-card-title class="text-h6">
            <v-icon class="mr-2">mdi-speedometer</v-icon>
            Realtime Processing Capacity
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="6" sm="3">
                <div class="text-caption">Overall Capacity</div>
                <div
                  class="text-h4"
                  :class="
                    realtimeCapacity >= (fileParameters?.numChannels ?? 0)
                      ? 'text-green'
                      : 'text-orange'
                  "
                >
                  {{ realtimeCapacity.toFixed(1) }}
                </div>
                <div class="text-caption">channels in realtime</div>
              </v-col>
              <v-col cols="6" sm="3">
                <div class="text-caption">Processing Speed</div>
                <div class="text-body-1">{{ samplesPerSecond?.toFixed(0) || 0 }} samples/sec</div>
                <div class="text-caption">
                  {{ ((realtimeCapacity / (fileParameters?.numChannels || 1)) * 100).toFixed(1) }}%
                  of file
                </div>
              </v-col>
              <v-col cols="6" sm="3">
                <div class="text-caption">File I/O Capacity</div>
                <div class="text-body-1">{{ readChannelCapacity?.toFixed(1) || 0 }} channels</div>
                <div class="text-caption">{{ readPercentage?.toFixed(1) || 0 }}% of time</div>
              </v-col>
              <v-col cols="6" sm="3">
                <div class="text-caption">Model Capacity</div>
                <div class="text-body-1">
                  {{ processChannelCapacity?.toFixed(1) || 0 }} channels
                </div>
                <div class="text-caption">{{ processPercentage?.toFixed(1) || 0 }}% of time</div>
              </v-col>
            </v-row>

            <!-- Time distribution visualization -->
            <div class="mt-3">
              <div class="text-caption mb-1">Time Distribution</div>
              <v-progress-linear height="20" class="rounded">
                <template v-slot:default>
                  <div class="d-flex w-100 h-100">
                    <div
                      class="d-flex align-center justify-center text-white text-caption"
                      :style="{
                        width: `${readPercentage || 0}%`,
                        backgroundColor: '#1976d2',
                        minWidth: (readPercentage || 0) > 10 ? 'auto' : '0px',
                      }"
                    >
                      {{ (readPercentage || 0) > 10 ? 'File I/O' : '' }}
                    </div>
                    <div
                      class="d-flex align-center justify-center text-white text-caption"
                      :style="{
                        width: `${processPercentage || 0}%`,
                        backgroundColor: '#388e3c',
                        minWidth: (processPercentage || 0) > 10 ? 'auto' : '0px',
                      }"
                    >
                      {{ (processPercentage || 0) > 10 ? 'Processing' : '' }}
                    </div>
                  </div>
                </template>
              </v-progress-linear>
              <div class="d-flex justify-space-between text-caption mt-1">
                <span style="color: #1976d2">
                  <v-icon size="small" color="#1976d2">mdi-circle</v-icon>
                  File I/O ({{ readPercentage?.toFixed(1) || 0 }}%)
                </span>
                <span style="color: #388e3c">
                  <v-icon size="small" color="#388e3c">mdi-circle</v-icon>
                  Processing ({{ processPercentage?.toFixed(1) || 0 }}%)
                </span>
              </div>
            </div>

            <!-- Overall efficiency bar -->
            <div class="mt-3">
              <div class="text-caption mb-1">Overall Efficiency</div>
              <v-progress-linear
                :model-value="Math.min(efficiency, 100)"
                :color="efficiency >= 100 ? 'green' : 'orange'"
                height="8"
                class="rounded"
              ></v-progress-linear>
              <div
                class="text-caption mt-1"
                :class="efficiency >= 100 ? 'text-green' : 'text-orange'"
              >
                {{ efficiency.toFixed(1) }}% ({{
                  efficiency >= 100 ? 'Can process in realtime' : 'Cannot process in realtime'
                }})
              </div>
            </div>
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
const processingMode = ref<'cpu' | 'gpu'>('cpu')
const gpuAvailable = ref(false)
const currentStatus = ref('')
const processingProgress = ref(0)

// New realtime capacity tracking
const realtimeCapacity = ref<number | null>(null)
const samplesPerSecond = ref<number | null>(null)
const readChannelCapacity = ref<number | null>(null)
const processChannelCapacity = ref<number | null>(null)
const readPercentage = ref<number | null>(null)
const processPercentage = ref<number | null>(null)

// Computed GPU status text
const gpuStatusText = computed(() => {
  if (gpuAvailable.value) {
    return 'GPU acceleration available (WebGPU or WebGL detected)'
  }
  return 'GPU acceleration not available - CPU processing only'
})

// Computed for backward compatibility
const useGPU = computed(() => processingMode.value === 'gpu')

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
  storageLayout: 'row-major' | 'column-major'
  isChunked: boolean
  chunkSize?: number[]
  fileFormat?: 'maxwell' | 'nwb' | 'unknown'
} | null>(null)

function handleFileSelected(files: File | File[]) {
  parsingFile.value = true
  const file = Array.isArray(files) ? files[0] : files
  if (file) {
    selectedFile.value = file
    fileParameters.value = null // Reset parameters
    // Clear all realtime capacity metrics when file changes
    realtimeCapacity.value = null
    samplesPerSecond.value = null
    readChannelCapacity.value = null
    processChannelCapacity.value = null
    readPercentage.value = null
    processPercentage.value = null
    // Send message to worker to extract h5 file parameters
    if (file.name.endsWith('.h5')) {
      worker.value?.postMessage({
        type: 'open',
        file: file,
      })
    }
  } else {
    selectedFile.value = null
    fileParameters.value = null
    parsingFile.value = false
    // Clear all realtime capacity metrics when no file selected
    realtimeCapacity.value = null
    samplesPerSecond.value = null
    readChannelCapacity.value = null
    processChannelCapacity.value = null
    readPercentage.value = null
    processPercentage.value = null
  }
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
      // Handle comprehensive realtime capacity updates
      realtimeCapacity.value = event.data.capacity
      samplesPerSecond.value = event.data.samplesPerSecond
      readChannelCapacity.value = event.data.readChannelCapacity
      processChannelCapacity.value = event.data.processChannelCapacity
      readPercentage.value = event.data.readPercentage
      processPercentage.value = event.data.processPercentage

      console.log(`📈 Realtime capacity: ${event.data.capacity.toFixed(1)} channels overall`)
      console.log(
        `   📁 File I/O: ${event.data.readChannelCapacity.toFixed(1)} channels (${event.data.readPercentage.toFixed(1)}%)`,
      )
      console.log(
        `   🧠 Processing: ${event.data.processChannelCapacity.toFixed(1)} channels (${event.data.processPercentage.toFixed(1)}%)`,
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
      // Do NOT clear realtimeCapacity and samplesPerSecond when stopped
      console.log('Processing stopped by user')
    } else if (event.data.type === 'result') {
      running.value = false
      errorMessage.value = null
      const result = event.data.result
      console.log('Received result:', result)
    } else if (event.data.type === 'error') {
      running.value = false
      errorMessage.value = event.data.error
      realtimeCapacity.value = null
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

  // Reset state but preserve realtime capacity display
  running.value = false
  processingProgress.value = 0
  currentStatus.value = 'Processing stopped'
  errorMessage.value = null
  // Do NOT clear realtimeCapacity and samplesPerSecond when stopping

  // Re-initialize worker for future use
  initializeWorkers()
}

function handleRun() {
  if (!selectedFile.value) {
    errorMessage.value = 'Please select a file to start.'
    return
  }

  // Clear all realtime capacity metrics when starting a new run
  realtimeCapacity.value = null
  samplesPerSecond.value = null
  readChannelCapacity.value = null
  processChannelCapacity.value = null
  readPercentage.value = null
  processPercentage.value = null

  if (selectedFile.value.name.endsWith('.h5')) {
    if (!fileParameters.value) {
      errorMessage.value = 'File parameters not extracted yet. Please wait or reselect the file.'
      return
    }
    // Use h5 workflow
    running.value = true
    errorMessage.value = null
    currentStatus.value = 'Starting spike detector with h5 file...'
    processingProgress.value = 0
    worker.value?.postMessage({
      type: 'run',
      file: selectedFile.value,
      modelsURL: `${window.location.href}models`,
      useGPU: useGPU.value,
    })
  } else {
    errorMessage.value = 'Unsupported file type. Please select a Maxwell raw .h5 file.'
  }
}

/**
 * Detect GPU availability in the browser
 */
async function detectGPUAvailability(): Promise<boolean> {
  // Check for WebGPU support first (best performance)
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter()
      if (adapter) {
        console.log('🚀 WebGPU available')
        return true
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
        console.log('🎮 WebGL available')
        return true
      }
    } catch (error) {
      console.warn('WebGL context creation failed:', error)
    }
  }

  console.log('⚠️ No GPU support detected')
  return false
}

// Fetch a sample file on application load
async function fetchSampleFile() {
  try {
    const sampleFileName = 'test_maxwell_raw.h5'
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
        type: 'open',
        file: file,
      })
    }
  } catch (error) {
    console.error('Error loading sample file:', error)
    currentStatus.value = 'Error loading sample file'
  }
}

onMounted(async () => {
  // Detect GPU availability first and set default processing mode
  gpuAvailable.value = await detectGPUAvailability()
  processingMode.value = gpuAvailable.value ? 'gpu' : 'cpu'

  initializeWorkers()
  await fetchSampleFile()
})
</script>

<style scoped>
/* Remove any custom styles that might be hiding the file input */
</style>
