import torch
import transformers

from ...utils.generate import generate_fast


CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(model, tok, length_params, device):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]
        # print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def find_sublist_start_index(list1, list2):
    for i in range(len(list1) - len(list2) + 1):
        if all(a == b for a, b in zip(list1[i:i + len(list2)], list2)):
            return i
    return None


def parent_module(model, pname):
    components = pname.split('.')
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


def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")


def tokenize(batch, tokenizer, device, context_templates, hparams):
    len_temp = len(context_templates)
    prompts = [item['prompt'] for item in batch]
    labels = [item['target_new'] for item in batch]
    loc_prompts = [item['loc_prompt'] for item in batch]

    mask_token = -100

    if getattr(hparams, "use_chat_template", False):
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": templ.format(p)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for templ in context_templates
            for p in prompts
        ]
        full_prompt = [
            f"{prompt_text} {label}"
            for prompt_text, label in zip(prompt_texts, labels * len(context_templates))
        ]
    else:
        prompt_texts = [
            f"{templ.format(p)}"
            for templ in context_templates
            for p in prompts
        ]
        full_prompt = [
            f"{templ.format(p + ' ' + l)}"
            for templ in context_templates
            for p, l in zip(prompts, labels)
        ]

    prompt_ids = tokenizer(
        prompt_texts,
        return_tensors="pt", padding=True, truncation=True,
    )["input_ids"]

    full_prompt += loc_prompts

    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    act_masks = []
    deact_masks = []

    for i, loc_prompt in enumerate(loc_prompts):
        if loc_prompt in prompts[i]:
            subject_token = tokenizer.encode(' ' + loc_prompt, add_special_tokens=False)
            subject_token1 = tokenizer.encode(loc_prompt, add_special_tokens=False)
            act_mask = torch.zeros_like(tokens['input_ids'][int(i * len_temp):int((i + 1) * len_temp)])
            deact_mask = torch.zeros_like(tokens['input_ids'][int(i * len_temp):int((i + 1) * len_temp)])
            for j, token in enumerate(tokens['input_ids'][int(i * len_temp):int((i + 1) * len_temp)]):
                token_ids = token.detach().cpu().numpy().tolist()
                start_idx = find_sublist_start_index(token_ids, subject_token)
                subject_length = len(subject_token)
                if start_idx is None:
                    start_idx = find_sublist_start_index(token_ids, subject_token1)
                    subject_length = len(subject_token1)
                if start_idx is None:
                    act_mask = None
                    deact_mask = None
                    break
                act_mask[j][start_idx: start_idx + subject_length] = 1
                deact_mask[j][:start_idx] = 1
                deact_mask[j][start_idx + subject_length:] = 1
        else:
            act_mask = None
            deact_mask = None

        act_masks.append(act_mask)
        deact_masks.append(deact_mask)

    act_masks = [mask.to(device) if mask is not None else None for mask in act_masks]
    deact_masks = [mask.to(device) if mask is not None else None for mask in deact_masks]
    tokens = {key: val.to(device) for key, val in tokens.items()}

    return tokens, act_masks, deact_masks
