#!/usr/bin/env python3
"""
GPU Availability Test Script
Tests and displays GPU information for CUDA/PyTorch setup.
"""

from typing import Dict, Any


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        Dict with keys:
            - 'available': bool, True if GPU is available
            - 'device': str, 'cuda' or 'cpu'
            - 'cuda_available': bool, True if CUDA is available
            - 'cuda_version': str, CUDA version or 'N/A'
            - 'device_name': str, GPU device name or 'N/A'
            - 'device_count': int, number of GPUs available
            - 'current_device': int, current GPU device index
            - 'torch_version': str, PyTorch version or 'N/A'
    """
    gpu_info = {
        'available': False,
        'device': 'cpu',
        'cuda_available': False,
        'cuda_version': 'N/A',
        'device_name': 'N/A',
        'device_count': 0,
        'current_device': -1,
        'torch_version': 'N/A'
    }
    
    try:
        import torch
        gpu_info['torch_version'] = torch.__version__
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if gpu_info['cuda_available']:
            gpu_info['available'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['cuda_version'] = torch.version.cuda or 'Unknown'
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARNING] Error checking GPU: {e}")
    
    return gpu_info


def print_gpu_info(gpu_info: Dict[str, Any] = None):
    """
    Print formatted GPU information to console.
    
    Args:
        gpu_info: Dict from check_gpu_availability(). If None, will check first.
    """
    if gpu_info is None:
        gpu_info = check_gpu_availability()
    
    print("\n" + "=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    print(f"GPU Available:       {gpu_info['available']}")
    print(f"Device:              {gpu_info['device'].upper()}")
    print(f"CUDA Available:      {gpu_info['cuda_available']}")
    print(f"CUDA Version:        {gpu_info['cuda_version']}")
    print(f"PyTorch Version:     {gpu_info['torch_version']}")
    if gpu_info['cuda_available']:
        print(f"GPU Device Name:     {gpu_info['device_name']}")
        print(f"Device Count:        {gpu_info['device_count']}")
        print(f"Current Device:      {gpu_info['current_device']}")
    print("=" * 60 + "\n")


def main():
    """Main entry point for GPU testing."""
    print("\n[*] Testing GPU Availability...")
    gpu_info = check_gpu_availability()
    print_gpu_info(gpu_info)
    
    if gpu_info['available']:
        print("[OK] GPU is available and ready to use!")
        return 0
    else:
        print("[WARNING] No GPU detected. Will run on CPU.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
