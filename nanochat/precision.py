"""
Precision utilities for nanochat.
Handles global precision settings via environment variable NANOCHAT_PRECISION.
Supported precisions: bfloat16, fp16, fp32, fp8
"""

import os
import torch
from typing import Optional, Union

# Global precision setting from environment variable
DEFAULT_PRECISION = "bfloat16"
PRECISION_ENV_VAR = "NANOCHAT_PRECISION"

def get_precision() -> str:
    """Get the current precision setting from environment variable."""
    return os.environ.get(PRECISION_ENV_VAR, DEFAULT_PRECISION).lower()

def get_torch_dtype(precision: Optional[str] = None) -> torch.dtype:
    """
    Convert precision string to torch.dtype.
    
    Args:
        precision: Precision string ('bfloat16', 'fp16', 'fp32', 'fp8'). 
                   If None, uses the global precision setting.
    
    Returns:
        torch.dtype: The corresponding torch dtype
        
    Raises:
        ValueError: If precision is not supported
    """
    if precision is None:
        precision = get_precision()
    
    precision_map = {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,  # alias
        'fp16': torch.float16,
        'float16': torch.float16,  # alias
        'fp32': torch.float32,
        'float32': torch.float32,  # alias
        'fp8': torch.float8_e4m3fn,  # PyTorch 2.1+ supports float8
    }
    
    if precision not in precision_map:
        supported = ', '.join(precision_map.keys())
        raise ValueError(f"Unsupported precision '{precision}'. Supported: {supported}")
    
    return precision_map[precision]

def get_autocast_dtype(precision: Optional[str] = None) -> torch.dtype:
    """
    Get the autocast dtype for the given precision.
    For fp8, falls back to fp16 for autocast compatibility.
    """
    if precision is None:
        precision = get_precision()
    
    # For autocast, fp8 is not supported, so we fall back to fp16
    if precision in ['fp8']:
        return torch.float16
    
    return get_torch_dtype(precision)

def convert_tensor_to_precision(tensor: torch.Tensor, precision: Optional[str] = None) -> torch.Tensor:
    """
    Convert a tensor to the specified precision.
    
    Args:
        tensor: Input tensor
        precision: Target precision. If None, uses global precision.
        
    Returns:
        torch.Tensor: Tensor converted to target precision
    """
    if precision is None:
        precision = get_precision()
    
    target_dtype = get_torch_dtype(precision)
    
    # Skip conversion if already in target dtype
    if tensor.dtype == target_dtype:
        return tensor
    
    # Handle special cases
    if precision in ['fp8'] and not hasattr(torch, 'float8_e4m3fn'):
        # Fallback to fp16 if fp8 is not supported
        target_dtype = torch.float16
        print(f"Warning: FP8 not supported in this PyTorch version, falling back to FP16")
    
    return tensor.to(dtype=target_dtype)

def set_precision(precision: str) -> None:
    """
    Set the global precision setting.
    
    Args:
        precision: Precision string to set
    """
    precision = precision.lower()
    # Validate precision by trying to get torch dtype
    get_torch_dtype(precision)
    os.environ[PRECISION_ENV_VAR] = precision

def print_precision_info() -> None:
    """Print current precision settings."""
    current = get_precision()
    torch_dtype = get_torch_dtype(current)
    autocast_dtype = get_autocast_dtype(current)
    
    print(f"Precision Settings:")
    print(f"  Environment Variable: {PRECISION_ENV_VAR}={current}")
    print(f"  Torch Dtype: {torch_dtype}")
    print(f"  Autocast Dtype: {autocast_dtype}")
    
    # Check if fp8 is available
    if current == 'fp8' and not hasattr(torch, 'float8_e4m3fn'):
        print(f"  Warning: FP8 requested but not available in this PyTorch version")