from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.devices import get_model_device, get_module_device
from ...utils import nethook
from ...utils import repr_tools


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: dict,
    hparams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: list[str],
) -> torch.Tensor:
    # print("Computing right vector (v)")
    input_device = get_model_device(model, fallback=hparams.device)
    delta_device = get_module_device(
        model,
        hparams.mlp_module_tmp.format(layer),
        fallback=input_device,
    )

    target_ids = tok(request["target_new"], return_tensors="pt").to(input_device)["input_ids"][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(input_device)

    rewriting_targets = torch.tensor(-100, device=input_device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)
    # print(f"Rewrite layer is {layer}")
    # print(f"Tying optimization objective to {loss_layer}")

    delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=delta_device)
    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer: str) -> Any:
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            hidden_state = nethook.get_hidden_state(cur_out)
            delta_for_hidden = delta.to(device=hidden_state.device, dtype=hidden_state.dtype)
            if target_init is None:
                # print("Recording initial value of v*")
                target_init = hidden_state[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(hidden_state):
                    hidden_state[idx, i, :] += delta_for_hidden
                else:
                    hidden_state[i, idx, :] += delta_for_hidden

            return nethook.replace_hidden_state(cur_out, hidden_state)

        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init).pow(2)
        )
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        # print(
        #     f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
        #     f"avg prob of [{request['target_new']}] "
        #     f"{torch.exp(-nll_loss_each).mean().item()}"
        # )
        if loss < 5e-2:
            break

        loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    if hparams.enable_random_prefix_keys:
        cur_inputs, cur_outputs = [], []
        for context_template in context_templates:
            cur_input, cur_output = get_module_input_output_at_word(
                model,
                tok,
                layer,
                context_template=context_template.format(request["prompt"]),
                word=request["subject"],
                module_template=hparams.rewrite_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )
            cur_inputs.append(cur_input)
            cur_outputs.append(cur_output)

        cur_input = torch.stack(cur_inputs).mean(0)
        cur_output = torch.stack(cur_outputs).mean(0)
        target = target_init + delta.to(device=target_init.device, dtype=target_init.dtype)
    else:
        cur_input, cur_output = get_module_input_output_at_word(
            model,
            tok,
            layer,
            context_template=request["prompt"],
            word=request["subject"],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
        target = cur_output + delta.to(device=cur_output.device, dtype=cur_output.dtype)

    cur_input = cur_input.to(target.device)
    cur_output = cur_output.to(target.device)
    left_vector = left_vector.to(cur_input.device)
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    # print(f"Delta norm: {(target - cur_output).norm().item()}")
    # print(
    #     f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    # )
    # print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    # print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_"):]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    input_prompt: str | None = None,
) -> int:
    ret = None
    if fact_token_strategy == "last":
        ret = len(tok.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return ret
