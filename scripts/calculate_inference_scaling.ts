#!/usr/bin/env node
/**
 * Standalone TypeScript script to calculate inference scaling from scaled_traces.bin
 * This should match the Python version and produce: 0.3761194029850746
 */

import * as fs from 'fs'
import * as path from 'path'

/**
 * Calculate linear interpolation for quantiles
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
 * Calculate inference scaling using the exact same logic as the Python version
 */
function calculateInferenceScaling(
  scaledTracesPath: string = 'public/models/scaled_traces.bin',
  inferenceScalingNumerator: number = 12.6,
  preMedianFrames: number = 1000
): number {
  console.log(`Loading scaled traces from: ${scaledTracesPath}`)
  
  // Load the binary file
  const buffer = fs.readFileSync(scaledTracesPath)
  
  // Create Float16Array from buffer - this is how it's loaded in the worker
  const scaledTraces = new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2)
  
  // Determine dimensions (942 channels x 2000 samples)
  const numChannels = 942
  const totalSamples = 2000
  
  console.log(`Scaled traces shape: [${numChannels}, ${totalSamples}]`)
  console.log(`Window will be: [${numChannels}, ${preMedianFrames}]`)
  
  // Extract initial window for each channel and convert to Float32 (matching Python)
  // This matches the exact logic: convert float16 to float32 for computation
  const window = new Float32Array(numChannels * preMedianFrames)
  
  for (let channel = 0; channel < numChannels; channel++) {
    const channelStart = channel * totalSamples
    for (let sample = 0; sample < preMedianFrames; sample++) {
      // Convert from float16 (stored as uint16) to float32
      const uint16Value = scaledTraces[channelStart + sample]
      // This is the float16 to float32 conversion
      window[channel * preMedianFrames + sample] = float16ToFloat32(uint16Value)
    }
  }
  
  console.log('Window extraction completed')
  
  // Calculate IQRs for each channel (matching PyTorch quantile method)
  const iqrs = new Float32Array(numChannels)
  
  for (let channel = 0; channel < numChannels; channel++) {
    // Extract channel data
    const channelData = new Float32Array(preMedianFrames)
    for (let i = 0; i < preMedianFrames; i++) {
      channelData[i] = window[channel * preMedianFrames + i]
    }
    
    // Sort for quantile calculation
    channelData.sort()
    
    // Calculate 25th and 75th percentiles using linear interpolation
    // This matches torch.quantile behavior
    const q25 = linearInterpolate(channelData, 0.25)
    const q75 = linearInterpolate(channelData, 0.75)
    
    // IQR = Q75 - Q25
    iqrs[channel] = q75 - q25
  }
  
  console.log(`IQRs shape: [${iqrs.length}]`)
  console.log('First 10 IQRs:', Array.from(iqrs.slice(0, 10)))
  
  // Calculate median of IQRs (matching PyTorch median)
  const medianIqr = calculateMedian(iqrs)
  console.log(`Median IQR: ${medianIqr}`)
  
  // Calculate inference scaling
  const inferenceScaling = medianIqr !== 0 ? inferenceScalingNumerator / medianIqr : 1
  
  console.log(`Inference scaling numerator: ${inferenceScalingNumerator}`)
  console.log(`Inference scaling: ${inferenceScaling}`)
  
  return inferenceScaling
}

/**
 * Convert float16 stored as uint16 to float32
 * This implements the IEEE 754 float16 to float32 conversion
 */
function float16ToFloat32(uint16Value: number): number {
  const sign = (uint16Value & 0x8000) >> 15
  const exponent = (uint16Value & 0x7c00) >> 10
  const mantissa = uint16Value & 0x03ff
  
  let result: number
  
  if (exponent === 0) {
    if (mantissa === 0) {
      // Zero
      result = 0
    } else {
      // Subnormal number
      result = Math.pow(2, -14) * (mantissa / 1024)
    }
  } else if (exponent === 0x1f) {
    if (mantissa === 0) {
      // Infinity
      result = Infinity
    } else {
      // NaN
      result = NaN
    }
  } else {
    // Normal number
    result = Math.pow(2, exponent - 15) * (1 + mantissa / 1024)
  }
  
  return sign ? -result : result
}

/**
 * Main function to test the calculation
 */
function main() {
  console.log('='.repeat(60))
  console.log('INFERENCE SCALING CALCULATION - TYPESCRIPT')
  console.log('='.repeat(60))
  
  // Test with public models data
  try {
    const result = calculateInferenceScaling('public/models/scaled_traces.bin')
    console.log(`\n✅ RESULT: ${result}`)
    
    // Check if it matches the expected value
    const expected = 0.3761194029850746
    console.log(`Expected: ${expected}`)
    console.log(`Match (4 decimal places): ${Math.abs(result - expected) < 1e-4}`)
    console.log(`Difference: ${Math.abs(result - expected)}`)
    
  } catch (error) {
    console.error('❌ Failed to calculate inference scaling:', error)
  }
}

// Run main if this file is executed directly
main()

export { calculateInferenceScaling }
