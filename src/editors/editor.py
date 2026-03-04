import torch

from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..hparams import HyperParams
from ..utils import EditRequest, EvalSuite, EvalResult


supported_model_families = ["gpt", "llama"]


class Editor(ABC):
    def __init__(self, hparams: HyperParams):
        model_name = hparams.model_name
        if not any([model_family in model_name
                    for model_family in supported_model_families]):
            raise ValueError(f"Unsupported model family: {model_name}.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if hparams.fp16 else torch.float32,
        ).to(f"cuda:{hparams.gpu_id}")  # type: ignore[arg-type]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.hparams = hparams

        self._weights_copy = None

    def __del__(self):
        del self.model

    def edit(self, edit_request: EditRequest, is_sequential=False):
        if not is_sequential:
            self._restore()
        self._apply(edit_request)

    # def edit_batch(self, edit_requests: list[EditRequest], is_sequential=False):
    #     if not is_sequential:
    #         self._restore()
    #     self._apply_batch(edit_requests)

    def eval(self, eval_suite: EvalSuite, eval_method: str) -> EvalResult:
        if eval_method not in ["teacher_forcing", "generation", "llm_judge"]:
            raise ValueError(f"Unknown evaluation method: {eval_method}.")
        pass

    def generate(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_lengths = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_lengths:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    @abstractmethod
    def _apply(self, edit_request: EditRequest): ...

    # @abstractmethod
    # def _apply_batch(self, edit_requests: list[EditRequest]):...

    @abstractmethod
    def _restore(self): ...
