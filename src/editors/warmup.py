from __future__ import annotations

import gc
from time import perf_counter

import torch


_WARMUP_MARKER = "_knowledge_surgeon_warmed_up"
_WARMUP_TEXT = "The capital city of France is"
_WARMUP_MAX_LENGTH = 64


def warm_up_model(model, tokenizer, *, model_key: str) -> bool:
    """Compile representative model forward/backward kernels without editing weights."""
    if getattr(model, _WARMUP_MARKER, False):
        return False
    if not torch.cuda.is_available():
        print(f"[model-warmup-skipped] model={model_key} reason=cuda_unavailable")
        return False

    embedding = model.get_input_embeddings()
    input_device = embedding.weight.device
    encoded = tokenizer(
        _WARMUP_TEXT,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=_WARMUP_MAX_LENGTH,
    )
    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_device)

    parameter_requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    module_training = {module: module.training for module in model.modules()}
    cuda_devices = {
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    }
    inputs_embeds = None
    outputs = None
    loss = None
    started = perf_counter()
    print(f"[model-warmup-start] model={model_key}", flush=True)
    try:
        # Gradients are needed only with respect to the input so model weights stay untouched.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        inputs_embeds = embedding(input_ids).detach().requires_grad_(True)
        with torch.enable_grad():
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss = outputs.logits[..., -1, :].float().mean()
            loss.backward()
        for device in sorted(cuda_devices, key=lambda value: value.index or 0):
            torch.cuda.synchronize(device)
        setattr(model, _WARMUP_MARKER, True)
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, requires_grad in parameter_requires_grad.items():
            parameter.requires_grad_(requires_grad)
        for module, training in module_training.items():
            module.training = training
        inputs_embeds = None
        outputs = None
        loss = None
        input_ids = None
        attention_mask = None
        encoded = None
        gc.collect()
        for device in sorted(cuda_devices, key=lambda value: value.index or 0):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    elapsed = perf_counter() - started
    print(f"[model-warmup-done] model={model_key} elapsed={elapsed:.2f}s", flush=True)
    return True
