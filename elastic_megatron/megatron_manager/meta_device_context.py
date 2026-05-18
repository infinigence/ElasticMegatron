"""
Meta Device Context Manager

This context manager redirects all device operations to 'meta' device:
- All tensor creation with device parameter uses 'meta'
- tensor.cuda() and tensor.cpu() calls are redirected to meta device
- torch.cuda.current_device() returns 'meta'
- nn.Module.cuda() moves module to meta device

Usage:
    with MetaDeviceContext():
        model, optimizer, scheduler = setup_model_and_optimizer(...)
        # All tensors will be on meta device, no GPU memory allocated
"""

from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch


def _normalize_device(device: Any) -> str:
    """Normalize device to string, converting cuda/cpu to meta."""
    if device is None:
        return "meta"
    if isinstance(device, str):
        if device.startswith("cuda") or device == "cpu":
            return "meta"
        return device
    if isinstance(device, torch.device):
        if device.type in ("cuda", "cpu"):
            return "meta"
        return str(device)
    if isinstance(device, int):
        # Device index like 0, 1, etc. -> meta
        return "meta"
    return "meta"


class MetaDeviceContext:
    """Context manager that redirects all device operations to 'meta' device."""

    def __init__(self):
        self._patched = {}

    def _patch_tensor_cuda(self):
        """Patch tensor.cuda() to return meta device tensors."""
        original_cuda = torch.Tensor.cuda

        @wraps(original_cuda)
        def patched_cuda(self_tensor, device=None):
            return self_tensor.to("meta")

        torch.Tensor.cuda = patched_cuda
        self._patched["torch.Tensor.cuda"] = original_cuda

    def _patch_tensor_cpu(self):
        """Patch tensor.cpu() to return meta device tensors."""
        original_cpu = torch.Tensor.cpu

        @wraps(original_cpu)
        def patched_cpu(self_tensor):
            return self_tensor.to("meta")

        torch.Tensor.cpu = patched_cpu
        self._patched["torch.Tensor.cpu"] = original_cpu

    def _patch_tensor_to(self):
        """Patch tensor.to() to redirect cuda/cpu to meta."""
        original_to = torch.Tensor.to

        @wraps(original_to)
        def patched_to(self_tensor, *args, **kwargs):
            # Handle .to(device) or .to(device=...)
            if "device" in kwargs:
                kwargs["device"] = _normalize_device(kwargs["device"])
            elif len(args) > 0:
                # First arg might be device, dtype, or both
                first_arg = args[0]
                if isinstance(first_arg, (str, torch.device, int)) or first_arg is None:
                    # It's a device
                    args = (_normalize_device(first_arg),) + args[1:]
                elif hasattr(first_arg, "type") and hasattr(first_arg, "index"):
                    # It's a torch.device object
                    args = (_normalize_device(first_arg),) + args[1:]
            return original_to(self_tensor, *args, **kwargs)

        torch.Tensor.to = patched_to
        self._patched["torch.Tensor.to"] = original_to

    def _patch_module_cuda(self):
        """Patch nn.Module.cuda() to move to meta device."""
        original_cuda = torch.nn.Module.cuda

        @wraps(original_cuda)
        def patched_cuda(self_module, device=None):
            return self_module.to("meta")

        torch.nn.Module.cuda = patched_cuda
        self._patched["torch.nn.Module.cuda"] = original_cuda

    def _patch_module_to(self):
        """Patch nn.Module.to() to redirect cuda/cpu to meta."""
        original_to = torch.nn.Module.to

        @wraps(original_to)
        def patched_to(self_module, *args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _normalize_device(kwargs["device"])
            elif len(args) > 0:
                first_arg = args[0]
                if isinstance(first_arg, (str, torch.device, int)) or first_arg is None:
                    args = (_normalize_device(first_arg),) + args[1:]
            return original_to(self_module, *args, **kwargs)

        torch.nn.Module.to = patched_to
        self._patched["torch.nn.Module.to"] = original_to

    def _patch_torch_function(self, func_name: str):
        """Patch torch functions like torch.empty, torch.zeros, etc. to use meta device."""
        if not hasattr(torch, func_name):
            return
        original_func = getattr(torch, func_name)

        @wraps(original_func)
        def patched_func(*args, **kwargs):
            # Replace device parameter with 'meta'
            if "device" in kwargs:
                kwargs["device"] = _normalize_device(kwargs["device"])
            return original_func(*args, **kwargs)

        setattr(torch, func_name, patched_func)
        self._patched[f"torch.{func_name}"] = original_func

    def _patch_device_constructor(self):
        """Patch torch.device() constructor to convert cuda/cpu to meta."""
        # torch.device is a class, we need to wrap it properly
        original_device_class = torch.device

        class PatchedDevice:
            """Wrapper for torch.device that converts cuda/cpu to meta."""

            def __new__(cls, device):
                normalized = _normalize_device(device)
                return original_device_class(normalized)

            # Preserve class attributes
            @staticmethod
            def __call__(device):
                normalized = _normalize_device(device)
                return original_device_class(normalized)

        # Copy class attributes
        for attr in dir(original_device_class):
            if not attr.startswith("_") and attr not in ["__new__", "__call__"]:
                try:
                    setattr(PatchedDevice, attr, getattr(original_device_class, attr))
                except (AttributeError, TypeError):
                    pass

        # Replace torch.device
        torch.device = PatchedDevice
        self._patched["torch.device"] = original_device_class

    def _patch_tensor_parameter_type(self):
        """Patch tensor.type() to return 'meta' device tensors."""
        original_tensor_type = torch.Tensor.type
        original_parameter_type = torch.nn.Parameter.type

        def get_patched_type(fn):
            @wraps(fn)
            def patched_fn(*args, **kwargs):
                res = fn(*args, **kwargs)
                if "meta" in res:
                    return res.replace("meta", "cuda")

            return patched_fn

        torch.Tensor.type = get_patched_type(original_tensor_type)
        torch.nn.Parameter.type = get_patched_type(original_parameter_type)

        self._patched["torch.Tensor.type"] = original_tensor_type
        self._patched["torch.nn.Parameter.type"] = original_parameter_type

    def __enter__(self):
        """Enter the context and apply patches."""
        # Patch tensor methods
        self._patch_tensor_cuda()
        self._patch_tensor_cpu()
        self._patch_tensor_to()

        # Patch module methods
        self._patch_module_cuda()
        self._patch_module_to()

        self._patch_tensor_parameter_type()

        # Patch torch functions that create tensors
        tensor_creation_funcs = [
            "empty",
            "zeros",
            "ones",
            "randn",
            "rand",
            "arange",
            "tensor",
            "as_tensor",
            "full",
            "linspace",
            "logspace",
            "eye",
            "empty_like",
            "zeros_like",
            "ones_like",
            "randn_like",
            "rand_like",
            "full_like",
        ]
        for func_name in tensor_creation_funcs:
            self._patch_torch_function(func_name)

        # Patch device constructor
        self._patch_device_constructor()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original functions."""
        for key, original_func in self._patched.items():
            parts = key.split(".")
            assert len(parts) >= 2 and parts[0] == "torch", (
                f"Invalid key in patched functions: {key}"
            )
            sub_obj = torch
            sub_idx = 1
            while sub_idx < len(parts) - 1:
                sub_obj = getattr(sub_obj, parts[sub_idx])
                sub_idx += 1
            setattr(sub_obj, parts[-1], original_func)
        self._patched.clear()


# Convenience function
@contextmanager
def meta_device_context():
    """Convenience function for MetaDeviceContext."""
    with MetaDeviceContext():
        yield
