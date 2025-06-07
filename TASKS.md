# Tasks
Remaining tasks to complete

## Agent Task List

## Task List: Convert rt_sort.py detect_sequences() to TypeScript + ONNX

### **Phase 1: Core Data Processing & Model Inference**

#### Task 1: **Validate Inference Scaling Computation**
- **Goal**: Ensure the existing TypeScript inference scaling matches Python implementation
- **Input**: `public/models/scaled_traces.bin` (942 channels × 2000 samples, float16)
- **Expected Output**: Compare TypeScript result with Python `inference_scaling_numerator / median(iqrs)`
- **Validation**: Create a small Python script that computes inference scaling from the same data and compare results
- **Files to modify**: `src/worker.ts` (already partially implemented)

#### Task 2: **Implement ONNX Model Detection Pipeline**
- **Goal**: Run the detection model on scaled traces to produce model outputs
- **Input**: `public/models/scaled_traces.bin` + computed inference_scaling
- **Expected Output**: Model predictions matching `public/models/model_outputs.bin` (942 × 1200, float16)
- **Implementation**:
  - Load `detect.onnx` model
  - Process traces in windows (like Python's `all_start_frames` loop)
  - Apply median subtraction and scaling (`traces_torch * input_scale * inference_scaling`)
- **Validation**: Compare TypeScript outputs with existing `model_outputs.bin`
- **Files to create**: `src/detection.ts`

#### Task 3: **Implement Threshold-based Spike Detection**
- **Goal**: Apply stringent and loose thresholds to model outputs to detect spikes
- **Input**: Model outputs from Task 2
- **Expected Output**: Arrays of spike detections with timestamps and probabilities
- **Implementation**:
  - Apply `stringent_thresh` (0.275) and `loose_thresh` (0.1)
  - Extract spike times and amplitudes above thresholds
- **Validation**: Compare detected spikes count and timing with Python implementation
- **Files to create**: `src/spike-detection.ts`

### **Phase 2: Sequence Formation & Clustering**

#### Task 4: **Implement Electrode Neighborhood Calculation**
- **Goal**: Calculate inner/outer electrode neighborhoods for each electrode
- **Input**: Channel locations from recording metadata
- **Expected Output**: Neighborhood maps (inner_radius=50μm, outer_radius=100μm)
- **Implementation**:
  - Parse electrode coordinates
  - Calculate Euclidean distances
  - Build neighborhood lookup tables
- **Validation**: Compare neighborhood counts with Python implementation
- **Files to create**: `src/electrode-neighborhoods.ts`

#### Task 5: **Implement Codetection Formation**
- **Goal**: Find codetections (spatiotemporally related spike detections)
- **Input**: Spike detections + electrode neighborhoods
- **Expected Output**: Grouped codetections within propagation windows
- **Implementation**:
  - Group detections by time windows (`ms_before=0.5ms`, `ms_after=0.5ms`)
  - Filter by spatial relationships (inner/outer electrodes)
  - Apply activity filters (`min_activity_root_cocs`, `min_activity_hz`)
- **Validation**: Compare codetection group counts and composition
- **Files to create**: `src/codetection.ts`

#### Task 6: **Implement Cluster Formation (form_all_clusters)**
- **Goal**: Form initial clusters from codetections
- **Input**: Codetections from Task 5
- **Expected Output**: Initial cluster objects with latency/amplitude distributions
- **Implementation**:
  - Group codetections by root electrode
  - Apply noise filtering (`min_elecs_for_array_noise`, `min_elecs_for_seq_noise`)
  - Create cluster data structures
- **Validation**: Compare cluster count and basic statistics with Python
- **Files to create**: `src/clustering.ts`

### **Phase 3: Advanced Clustering & Refinement**

#### Task 7: **Implement Gaussian Mixture Model Splitting**
- **Goal**: Split clusters based on latency and amplitude distributions
- **Input**: Initial clusters from Task 6
- **Expected Output**: Refined clusters after GMM-based splitting
- **Implementation**:
  - Port latency-based splitting (`max_n_components_latency=4`)
  - Port amplitude-based splitting (`max_n_components_amp=4`)
  - Apply minimum cluster size filters (`min_coc_n`, `min_coc_p`)
- **Validation**: Compare final cluster count and composition
- **Files to create**: `src/gmm-splitting.ts`
- **Note**: May need to use a JS/TS GMM library or implement simplified version

#### Task 8: **Implement Spike Reassignment**
- **Goal**: Reassign individual spikes to final clusters
- **Input**: Refined clusters + original spike detections
- **Expected Output**: Spike-to-cluster assignments
- **Implementation**:
  - Calculate spike-cluster similarity (latency + amplitude metrics)
  - Apply assignment thresholds (`max_latency_diff_spikes`, `max_amp_median_diff_spikes`)
  - Handle loose detection assignments (`min_loose_elec_prob`)
- **Validation**: Compare spike assignment counts per cluster
- **Files to create**: `src/spike-assignment.ts`

#### Task 9: **Implement Cluster Merging**
- **Goal**: Merge similar clusters (intra + inter merging)
- **Input**: Clusters with assigned spikes
- **Expected Output**: Final merged clusters
- **Implementation**:
  - Intra-merge: merge clusters from same root electrode
  - Inter-merge: merge clusters across different electrodes
  - Apply merging criteria (`max_latency_diff_sequences`, `max_amp_median_diff_sequences`)
- **Validation**: Compare final cluster count after merging
- **Files to create**: `src/cluster-merging.ts`

### **Phase 4: Integration & Final Output**

#### Task 10: **Implement RTSort Object Creation**
- **Goal**: Create final RTSort-equivalent object with all sequence data
- **Input**: Final merged clusters
- **Expected Output**: Structured object containing sequences and metadata
- **Implementation**:
  - Port RTSort class structure to TypeScript
  - Include sequence timing, root electrodes, spike assignments
  - Add utility methods (get_units, get_seq_root_elecs, etc.)
- **Validation**: Compare sequence counts and basic properties
- **Files to create**: `src/rt-sort.ts`

#### Task 11: **Implement Main detect_sequences() Function**
- **Goal**: Orchestrate the full pipeline in a single function
- **Input**: Scaled traces data (simulating recording input)
- **Expected Output**: Complete RTSort object or spike train arrays
- **Implementation**:
  - Chain all previous tasks in correct order
  - Handle error cases and edge conditions
  - Add progress reporting for web worker
- **Validation**: End-to-end comparison with Python output
- **Files to create**: `src/detect-sequences.ts`

#### Task 12: **Web Worker Integration & UI**
- **Goal**: Integrate full pipeline into web worker with progress reporting
- **Input**: Model files + data from UI
- **Expected Output**: Results posted back to main thread
- **Implementation**:
  - Update worker.ts to call detect_sequences()
  - Add progress callbacks for each phase
  - Handle memory management for large datasets
- **Validation**: Test full pipeline in browser environment
- **Files to modify**: `src/worker.ts`, `src/App.vue`

### **Testing & Validation Strategy**

Each task should include:
1. **Unit tests** comparing TypeScript outputs with Python reference data
2. **Performance benchmarks** to ensure reasonable execution time
3. **Memory usage monitoring** especially for large electrode arrays
4. **Edge case testing** with minimal/maximal parameter values

### **Available Test Data**
- `public/models/scaled_traces.bin` - Input ephys data (942 × 2000, float16)
- `public/models/model_outputs.bin` - Expected model outputs (942 × 1200, float16)
- Can generate additional Python reference data for intermediate steps as needed

This task list builds the pipeline incrementally, with each task producing a testable component that can be validated against the Python implementation before proceeding to the next step.
