from collections.abc import Mapping
from typing import Any

import torch


def get_model_device(model: torch.nn.Module, fallback: Any | None = None) -> torch.device:
    """Return the device that should receive model inputs."""
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device

    for buffer in model.buffers():
        if buffer.device.type != "meta":
            return buffer.device

    if fallback is not None:
        return torch.device(fallback)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_parameter_device(model: torch.nn.Module, name: str) -> torch.device:
    for param_name, param in model.named_parameters():
        if param_name == name:
            return param.device
    raise LookupError(name)


def get_module_device(
    model: torch.nn.Module,
    name: str,
    fallback: Any | None = None,
) -> torch.device:
    for module_name, module in model.named_modules():
        if module_name != name:
            continue
        for param in module.parameters(recurse=True):
            if param.device.type != "meta":
                return param.device
        for buffer in module.buffers(recurse=True):
            if buffer.device.type != "meta":
                return buffer.device
        break

    if fallback is not None:
        return torch.device(fallback)
    return get_model_device(model)


def move_to_device(data: Any, device: Any) -> Any:
    """Move tensors or nested tensor containers to a device."""
    target = torch.device(device)
    if isinstance(data, torch.Tensor):
        return data.to(target)
    if isinstance(data, Mapping):
        return type(data)({key: move_to_device(value, target) for key, value in data.items()})
    if isinstance(data, tuple):
        return tuple(move_to_device(value, target) for value in data)
    if isinstance(data, list):
        return [move_to_device(value, target) for value in data]
    return data


def to_param_device(
    tensor: torch.Tensor,
    param: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = dtype or tensor.dtype
    if tensor.device == param.device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(device=param.device, dtype=target_dtype)
