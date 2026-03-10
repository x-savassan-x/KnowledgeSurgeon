def get_parent(model, layer: str):
    parts = layer.split(".")
    parent_path = ".".join(parts[:-1])
    child_name = parts[-1]

    parent = model.get_submodule(parent_path)

    if not hasattr(parent, child_name):
        raise AttributeError(f"Module {parent_path} does not have attribute {child_name}")

    return parent


def tokenize(request, tokenizer, device, test=False):
    prompt, label = request.prompt, request.new_ans
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100 # ignore_index of CrossEntropyLoss

    if test or not label:
        tokens = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    else:
        full_prompt = [f"{p} {lab}" for p, lab in zip(prompt, label)]
        prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
        num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
        tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        for i in range(len(prompt)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    
    tokens = {f"{k1}" : v1.to(device) for k1, v1 in tokens.items()}
    return tokens
