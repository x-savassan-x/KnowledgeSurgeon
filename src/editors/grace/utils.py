from typing import Any

import torch


def parent_module(model, pname: str) -> Any:
    components = pname.split(".")
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent


def brackets_to_periods(name: str) -> str:
    return name.replace("[", ".").replace("]", "")


def tokenize(
    batch: dict[str, Any],
    tokenizer,
    device: torch.device | str,
    test: bool = False,
) -> dict[str, torch.Tensor]:
    prompt, label = batch["prompt"], batch["target_new"]
    if not isinstance(prompt, list):
        prompt = [prompt]
    if not isinstance(label, list):
        label = [label]
    mask_token = -100

    if test or not label:
        tokens = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    else:
        full_prompt = [f"{p} {lbl}" for p, lbl in zip(prompt, label)]
        prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
        num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
        tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        for i in range(len(prompt)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    tokens = {f"{k1}": v1.to(device) for k1, v1 in tokens.items()}
    return tokens
