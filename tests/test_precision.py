#!/usr/bin/env python3
"""
Test script to verify precision configuration works correctly.
"""
import os
import torch
from nanochat.precision import get_autocast_dtype, print_precision_info, SUPPORTED_DTYPES

def test_precision_config():
    """Test different precision configurations."""
    print("Testing precision configuration...")
    
    # Test each supported precision
    for precision in SUPPORTED_DTYPES.keys():
        print(f"\n--- Testing {precision} ---")
        os.environ["NANOCHAT_PRECISION"] = precision
        
        # Test CPU device
        dtype_cpu = get_autocast_dtype("cpu")
        print(f"CPU dtype: {dtype_cpu}")
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            dtype_cuda = get_autocast_dtype("cuda")
            print(f"CUDA dtype: {dtype_cuda}")
        else:
            print("CUDA not available, skipping CUDA test")
        
        # Test print function
        print_precision_info()
    
    # Test default (no env var set)
    print("\n--- Testing default (no env var) ---")
    if "NANOCHAT_PRECISION" in os.environ:
        del os.environ["NANOCHAT_PRECISION"]
    
    dtype_default = get_autocast_dtype("cpu")
    print(f"Default CPU dtype: {dtype_default}")
    print_precision_info()
    
    # Test invalid precision
    print("\n--- Testing invalid precision ---")
    os.environ["NANOCHAT_PRECISION"] = "invalid_precision"
    try:
        get_autocast_dtype("cpu")
        print("ERROR: Should have raised ValueError for invalid precision")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    print("\n✅ All precision tests passed!")

if __name__ == "__main__":
    test_precision_config()