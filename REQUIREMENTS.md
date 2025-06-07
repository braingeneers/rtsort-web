# RTSort Web Application - Project Requirements & Goals

## Project Overview
Single page web application that detects and plots spikes in ephys data streamed from files on the local machine running entirely in the browser.

## References
The RTSort [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312438) documents the spike detection and sorting algorithm.

Braindance, a sub-repository within this project under braindance/ contains an implementation of RTSort in the file:
```
braindance/braindance/core/spikesorter/rt_sort.py.
```

Specifically this web application should implement detect_sequences() in the browser.

## Technology Stack
- **Data Processing**: h5wasm for HDF5 file reading
- **ML Inference**: ONNX Runtime Web for RTSort model
- **Frontend Framework**: Vue 3 with Composition API
- **UI Library**: Vuetify 3 (Material Design components)
- **Build Tool**: Vite with TypeScript support
- **Styling**: Material Design + custom scientific visualization themes
- **Deployment**: Static site (GitHub Pages compatible)

## Style
- When possible keep the code concise, easy to read and understand. This is not a large scale consumer application so we don't need to account for every edge case and check every result. We can add that in later. For now we need to focus on getting the results concordant with the original python (rt_sort.py) and the published paper. Then we need to optimize it to run in realtime.
- Where possible avoid adding extraneous libraries or packages unless they are widely used and would greatly reduce the code size.

## Primary Goals

### Core Functionality
- **Real-time spike detection**: Process ephys data from .h5 files with minimal latency using RTSort algorithm
- **Interactive visualization**: Provide responsive, zoomable plots of detected spikes with smooth user interactions
- **Browser-based processing**: No server-side dependencies - all data processing happens client-side for privacy and performance
- **File format support**: Native .h5 file reading in the browser using h5wasm

### User Experience
- **Drag-and-drop file loading**: Intuitive file upload interface with clear feedback
- **Material Design**: Clean, scientific aesthetic using Vuetify components
- **Responsive design**: Works seamlessly on desktop and tablet devices
- **Performance**: Smooth interactions even with large ephys datasets
- **Progressive disclosure**: Advanced features available but not overwhelming for basic use

### Technical Excellence
- **Modern Vue 3**: Composition API, reactive data handling, and component architecture
- **TypeScript safety**: Full type coverage for maintainable, bug-free code
- **WebAssembly integration**: Seamless RTSort algorithm execution via ONNX Runtime
- **Memory efficiency**: Handle multi-channel ephys datasets without browser crashes
- **Error resilience**: Graceful degradation and informative error messages

## Technical Requirements

### Browser Compatibility
- Chrome 90+, Safari 14+ (WebAssembly + ES2020 support)
- SharedArrayBuffer support preferred for threading
- File System Access API with fallback to traditional file input
- WebGL support for hardware-accelerated visualization

### Performance Targets
- Load typical 100MB .h5 ephys files within 5 seconds
- Spike detection processing at >1000 samples/second per channel
- UI remains responsive during background processing (Web Workers)
- Memory usage <2GB for datasets with 64+ channels

## Functional Requirements

### Data Input & Validation
- Support standard ephys .h5 file structures (MEA recordings, Intan, etc.)
- Validate file format and show descriptive error messages
- Handle large channel count recordings with common layouts.
- Support detecting spikes on a portion of the channels when processing all of them in realtime is not feasible.
- Support sampling rates from 1kHz to 30kHz
- Preview file metadata before full processing

### Spike Detection Pipeline
- Load and execute RTSort ONNX model for spike detection
- Configurable detection parameters (threshold, window size, etc.)
- Real-time processing with progress indicators
- Background processing using Web Workers to maintain UI responsiveness
- Export detected spike timestamps and waveforms

### Data Visualization
- **Multi-channel display**: Scrollable/zoomable waveform viewer
- **Spike overlays**: Detected spikes highlighted on raw data
- **Navigation controls**: Time-based scrubbing, jump-to-spike functionality
- **Statistics panel**: Spike rates, detection summary, channel health
- **Export capabilities**: Screenshots, data downloads, analysis reports

### User Interface Components
- **File upload area**: Drag-and-drop with file validation feedback
- **Control panel**: Detection parameters, visualization settings
- **Progress tracking**: Real-time processing status and ETA
- **Results dashboard**: Summary statistics and key findings
- **Settings**: User preferences, export options, advanced parameters

## Non-Functional Requirements

### Security & Privacy
- **Client-side only**: No data ever leaves the user's browser
- **Secure file handling**: Safe processing of user-uploaded files
- **Content Security Policy**: Prevent XSS and other web attacks
- **Memory safety**: Proper cleanup to prevent memory leaks

### Performance & Scalability
- **Lazy loading**: Load components and dependencies on demand
- **Web Workers**: CPU-intensive tasks don't block UI thread
- **Memory management**: Efficient handling of large datasets
- **Caching**: Smart caching of processed results and model weights

### Maintainability & Testing
- **TypeScript coverage**: 100% type safety across codebase
- **Component testing**: Unit tests for Vue components
- **Integration testing**: End-to-end spike detection workflows
- **Code documentation**: Clear JSDoc comments for complex algorithms
- **Modular architecture**: Easy to extend with new features

### Deployment & Distribution
- **Static site deployment**: Works with GitHub Pages, Netlify, Vercel
- **Build optimization**: Tree shaking, code splitting, asset optimization
- **Progressive Web App**: Offline capability for repeated use
- **Cross-platform**: Consistent experience across operating systems

## Success Criteria
- Successfully processes representative MEA and ephys datasets
- Neuroscience researchers can complete full analysis workflow in <5 minutes
- Application loads and becomes interactive in <3 seconds
- Zero data corruption or loss during processing
- Positive user feedback from target scientific community
- Deployment at https://braingeneers.github.io/rtsort-web/ is stable

## Future Considerations & Extensions
- **Optimization**: Part of the spike detection computation is external to the pytorch model. For example the median computation for each window processed. Where possible push these computations into ONNX by generating a new graph to export in python. Alternately we may convert some of the processing to Rust and export as WASM for performance - specifically computations that don't fit well into ONNX operators.
- **Batch processing**: Handle multiple files simultaneously
- **Real-time streaming**: Process live data feeds from hardware

## Development Guidelines for AI Agents
When working on this project, prioritize:
1. **User experience**: Every feature should feel intuitive to neuroscience researchers
2. **Performance**: Large datasets are common - optimize for memory and speed
3. **Scientific accuracy**: Spike detection results must be reliable and reproducible
4. **Code quality**: Maintain TypeScript safety and Vue 3 best practices
5. **Browser compatibility**: Test across target browsers, especially WebAssembly features
