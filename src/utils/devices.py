import torch


def get_model_device(model: torch.nn.Module, fallback=None) -> torch.device:
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


def get_module_device(
    model: torch.nn.Module,
    name: str,
    fallback=None,
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


def to_param_device(
    tensor: torch.Tensor,
    param: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = dtype or tensor.dtype
    if tensor.device == param.device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(device=param.device, dtype=target_dtype)


def copy_to_param(param: torch.Tensor, value) -> None:
    """Copy a tensor-like value into a parameter using the parameter's dtype/device."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    param[...] = value.to(device=param.device, dtype=param.dtype)
