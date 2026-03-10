import torch

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EditRequest:
    subject: str
    prompt: str
    old_ans: str
    new_ans: str


def load_model_and_tokenizer(model_name: str, fp16: bool, gpu_id: int, **_):
    if model_name.split("/")[-1].lower() not in (
        "gpt2-xl",
        "gpt-j-6b",
        "qwen2.5-7b",
        "meta-llama-3-8b"
    ):
        raise ValueError(f"Unsupported model: {model_name}.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if fp16 else torch.float32,
    ).cuda(gpu_id)  # type: ignore[arg-type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
