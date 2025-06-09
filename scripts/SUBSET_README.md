# Maxwell H5 Subset Tool

## Overview

This tool creates subsets of Maxwell MEA h5 files with fewer channels and shorter duration while preserving all essential metadata needed for analysis.

## Files Created

- `scripts/subset_maxwell_h5.py` - Main script for creating subsets
- `demo_subset.py` - Demonstration script showing different usage examples
- `test_h5py_access.py` - Validation script for h5py/h5wasm compatibility

## Usage

### Command Line

```bash
# Basic usage with defaults (8 channels, 5 seconds)
python scripts/subset_maxwell_h5.py data/MEA_rec_patch_ground_truth_cell7.raw.h5

# Custom parameters
python scripts/subset_maxwell_h5.py data/MEA_rec_patch_ground_truth_cell7.raw.h5 \
    --channels 8 \
    --seconds 5 \
    --output public/sample_maxwell_raw.h5 \
    --validate

# Help
python scripts/subset_maxwell_h5.py --help
```

### Python API

```python
from scripts.subset_maxwell_h5 import subset_maxwell_h5, validate_subset

# Create subset
output_file = subset_maxwell_h5(
    input_file="data/MEA_rec_patch_ground_truth_cell7.raw.h5",
    num_channels=8,
    num_seconds=5,
    output_file="data/my_subset.h5"
)

# Validate the result
validate_subset("data/MEA_rec_patch_ground_truth_cell7.raw.h5", output_file)
```

## What's Preserved

The subset files maintain all essential data structures:

- **Signal data**: Raw uint16 values for the first N channels and M seconds
- **Settings**: Gain, LSB (least significant bit), and high-pass filter settings
- **Mapping**: Electrode channel mappings and spatial coordinates
- **Time**: Time stamps (if present)
- **Version**: File format version
- **Metadata**: Other Maxwell-specific metadata

## Validation

The script includes comprehensive validation:

1. **Data integrity**: First 10 samples match between original and subset
2. **Parameter preservation**: Gain, LSB, and other settings are identical
3. **SpikeInterface compatibility**: Subset can be opened with SpikeInterface
4. **h5py/h5wasm compatibility**: Direct access works for JavaScript integration

## Example Results

From the test with `MEA_rec_patch_ground_truth_cell7.raw.h5`:

```
Original file: 942 channels, 3,723,600 samples (186.18s at 20kHz), 6.53 GiB
Subset file: 8 channels, 100,000 samples (5.0s at 20kHz), 1.53 MiB

First sample verification:
- Original: 513 (raw) → 3228.95025 μV (physical)
- Subset: 513 (raw) → 3228.95025 μV (physical)
- ✓ Values match exactly
```

## File Sizes

Different subset examples and their file sizes:

- 4 channels, 2 seconds: 0.31 MB
- 8 channels, 5 seconds: 1.53 MB  
- 16 channels, 10 seconds: 6.11 MB

Original file: 6.53 GB → 99.8% size reduction for default subset!

## JavaScript/h5wasm Usage

The subset files are fully compatible with h5wasm for browser-based analysis:

```javascript
// Example of how to read in JavaScript with h5wasm
const file = new h5wasm.File(buffer, "r");
const sig = file.get("sig");
const settings = file.get("settings");

const gain = settings.get("gain").value[0];
const lsb = settings.get("lsb").value[0];
const rawData = sig.value;

// Convert to physical values (microvolts)
const physicalData = rawData.map(val => val * lsb * 1e6 * gain);
```

This tool makes Maxwell MEA data much more manageable for web-based analysis and development!
