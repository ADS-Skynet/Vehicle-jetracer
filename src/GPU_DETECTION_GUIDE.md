# GPU Detection and Optimization Guide

## Overview

This document describes the GPU detection and optimization improvements made to the YOLO processor system. Three main enhancements have been implemented to make GPU availability checking easier and more reliable.

## Components Added

### 1. **test_gpu.py** - Standalone GPU Test Script

**Location:** `/home/siwoo/ADS-Skynet/vehicle/src/test_gpu.py`

A simple, standalone script to check GPU availability and display detailed CUDA/PyTorch information.

**Usage:**
```bash
python3 test_gpu.py
```

**Output Example:**
```
[*] Testing GPU Availability...

============================================================
GPU AVAILABILITY CHECK
============================================================
GPU Available:       False
Device:              CPU
CUDA Available:      False
CUDA Version:        N/A
PyTorch Version:     2.9.1+cpu
============================================================
```

**Features:**
- Detects CUDA availability
- Shows PyTorch version
- Displays GPU device name (if available)
- Shows device count and current device index
- Returns exit code 0 if GPU available, 1 if CPU only

### 2. **yolo_processor.py** - Enhanced YOLO Image Processor

**Location:** `/home/siwoo/ADS-Skynet/vehicle/src/yolo_processor.py`

Updated with three new GPU detection functions and integrated GPU checking at startup.

#### New Functions:

**`check_gpu_availability() -> Dict[str, Any]`**
- Returns comprehensive GPU information as a dictionary
- Safe error handling (catches ImportError if torch not installed)
- Returns all keys even if GPU not available

**Returns dictionary with keys:**
- `available`: bool - True if GPU is available
- `device`: str - 'cuda' or 'cpu'
- `cuda_available`: bool - True if CUDA is available
- `cuda_version`: str - CUDA version string
- `device_name`: str - GPU device name (e.g., "NVIDIA A100")
- `device_count`: int - Number of GPUs available
- `current_device`: int - Current GPU device index
- `torch_version`: str - PyTorch version string

**`print_gpu_info(gpu_info: Dict[str, Any] = None)`**
- Prints formatted GPU information to console
- Auto-checks GPU if gpu_info not provided
- Displays in easy-to-read table format with 60-character border

**`get_optimal_device(preferred_device: Optional[str] = None) -> str`**
- Intelligent device selection with fallback logic
- If preferred device specified:
  - Returns it if compatible with available hardware
  - Falls back to CPU if CUDA requested but not available
- If no preference: auto-selects GPU if available, else CPU
- Prints warnings if fallback occurs

#### Usage in yolo_processor.py:

**At Startup:**
```python
# Show GPU info at startup (unless --quiet flag)
if not args.quiet:
    gpu_info = check_gpu_availability()
    print_gpu_info(gpu_info)
```

**With --show-gpu Flag:**
```bash
python3 yolo_processor.py --show-gpu
```
Displays GPU information and exits without processing images.

**Automatic Device Selection:**
```python
# Automatically selects optimal device
device = get_optimal_device(args.gpu)
detector = YOLOv8Detector(weights_path, conf_thres=args.conf, device=device)
```

#### Full Usage Examples:

1. **Show GPU info only:**
   ```bash
   python3 yolo_processor.py --show-gpu
   ```

2. **Process images with auto GPU selection:**
   ```bash
   python3 yolo_processor.py image_dir/ -o output/
   ```

3. **Force CPU usage:**
   ```bash
   python3 yolo_processor.py image_dir/ --gpu cpu -o output/
   ```

4. **Use specific GPU (cuda:0):**
   ```bash
   python3 yolo_processor.py image_dir/ --gpu cuda:0 -o output/
   ```

5. **Suppress GPU info at startup:**
   ```bash
   python3 yolo_processor.py image_dir/ -q -o output/
   ```

## Key Features

### Automatic Fallback
- If user requests CUDA but it's not available, automatically falls back to CPU
- Prints warning message: `[WARNING] CUDA requested but not available. Falling back to CPU.`

### Error Handling
- Safely handles missing PyTorch installation
- Gracefully handles CUDA initialization errors
- Returns sensible defaults (CPU, N/A) if checks fail

### Integration
- GPU check happens automatically at startup (unless `--quiet` flag used)
- Device selection integrated into both YOLOv8Detector and standard detector initialization
- No impact on existing code - fully backward compatible

## Output Information

The GPU check displays:
```
============================================================
GPU AVAILABILITY CHECK
============================================================
GPU Available:       [True/False]
Device:              [CUDA/CPU]
CUDA Available:      [True/False]
CUDA Version:        [version or N/A]
PyTorch Version:     [version]
GPU Device Name:     [device name - only if CUDA available]
Device Count:        [number - only if CUDA available]
Current Device:      [index - only if CUDA available]
============================================================
```

## Current System Status

Based on the test results:
- **GPU Available:** ❌ No
- **CUDA Available:** ❌ No
- **PyTorch Version:** 2.9.1+cpu (CPU-only build)
- **Device Mode:** CPU

The system will run inference on CPU. To enable GPU support:
1. Install a GPU-enabled PyTorch build: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. Ensure NVIDIA CUDA Toolkit is installed
3. Run `python3 test_gpu.py` to verify

## Technical Details

### CUDA Detection Method
The code uses PyTorch's native CUDA checking:
```python
import torch
torch.cuda.is_available()  # Returns True if CUDA is available
torch.cuda.device_count()  # Number of GPUs
torch.cuda.get_device_name(0)  # GPU name
```

### Device Specifications Returned
When GPU is available, the dictionary includes:
- Exact CUDA version (e.g., "12.1")
- GPU model name (e.g., "NVIDIA A100 40GB")
- Number of GPUs in system
- Currently selected device index

### Backward Compatibility
- All existing code continues to work unchanged
- New GPU utilities are optional additions
- No breaking changes to function signatures
- Graceful degradation if torch not installed

## Files Modified/Created

| File | Type | Purpose |
|------|------|---------|
| `test_gpu.py` | Created | Standalone GPU test utility |
| `yolo_processor.py` | Enhanced | Added GPU detection functions |

## Integration Points

The GPU utilities can be easily integrated into other scripts:

```python
from yolo_processor import check_gpu_availability, print_gpu_info, get_optimal_device

# Check GPU
gpu_info = check_gpu_availability()

# Display info
print_gpu_info(gpu_info)

# Get optimal device
device = get_optimal_device(preferred_device='cuda:0')
```

## Performance Implications

- **GPU Processing:** ~10-100x faster for YOLO inference depending on model size
- **CPU Processing:** Slower but functional, suitable for testing/debugging
- **Memory:** GPU memory usage depends on model size and batch size

## Troubleshooting

**Q: PyTorch installed but CUDA not detected?**
- A: Install GPU-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cuXXX`

**Q: How do I force CPU-only mode?**
- A: Use `--gpu cpu` flag or run `CUDA_VISIBLE_DEVICES="" python3 yolo_processor.py`

**Q: System hangs during GPU detection?**
- A: This is unusual. Check PyTorch/CUDA installation. Can temporarily disable with `--quiet` flag.

## Summary

These GPU detection improvements provide:
1. ✅ Easy GPU availability checking
2. ✅ Automatic device selection with intelligent fallback
3. ✅ Detailed diagnostic information
4. ✅ No breaking changes to existing code
5. ✅ Simple standalone test script
