import torch

from functools import partial

from .eval_utils import (
    EvalSuite, EvalResult,
    evaluate_pairs, clean_response
)


def evaluate(model, tokenizer, eval_suite: EvalSuite) -> EvalResult:
    get_model_responses = partial(get_responses, model, tokenizer)

    result = EvalResult()

    if eval_suite.reliability is not None:
        eval_pairs = eval_suite.reliability
        result.reliability = evaluate_pairs(eval_pairs, get_model_responses)

    if eval_suite.locality is not None:
        result.locality = {loc_type: evaluate_pairs(eval_pairs, get_model_responses)
                           for loc_type, eval_pairs in eval_suite.locality.items()}

    if eval_suite.portability is not None:
        result.portability = {port_type: evaluate_pairs(eval_pairs, get_model_responses)
                              for port_type, eval_pairs in eval_suite.portability.items()}

    return result


def get_responses(model, tokenizer, prompts: list[str]) -> list[str]:
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.25,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    tokenizer.padding_side = orig_pad_side
    return [clean_response(response) for response in responses]
