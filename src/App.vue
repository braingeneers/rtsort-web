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

function initializeWorkers() {
  console.log('Initializing workers...')
  running.value = true
  worker.value = new SIMSWorker()
  worker.value.onmessage = (event) => {
    if (event.data.type === 'result') {
      running.value = false
      const result = event.data.result
      console.log('Received result:', result)
      console.log(`Expected value: 0.3761194050`)
      console.log(
        `Match: ${
          Math.abs(result.inferenceScaling - 0.376119405) < 1e-6 ? '✅ PERFECT' : '❌ FAILED'
        }`
      )
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
