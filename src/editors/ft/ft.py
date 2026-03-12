import torch
from torch.nn import CrossEntropyLoss

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import FtConfig


class FT(Editor):
    def __init__(self, config: FtConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "left"
        self._initial_weights: dict[str, torch.Tensor] = {}

    def reset(self):
        if not self._initial_weights:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._initial_weights.items():
                params[name][...] = weight.clone()

    def _apply(self, request: EditRequest):
        device = torch.device(f"cuda:{self.config.gpu_id}")
        target = " " + request.new_ans  # space required for correct tokenization

        # Find target weights
        weights = {
            n: p
            for n, p in self.model.named_parameters()
            for layer in self.config.layers
            if self.config.rewrite_module_tmp.format(layer) in n
        }

        # Save initial weights for reset (first call only)
        if not self._initial_weights:
            self._initial_weights = {k: v.detach().clone() for k, v in weights.items()}

        # Save pre-edit weights for norm clamping
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}

        # Enable gradient only for target weights
        for name, w in self.model.named_parameters():
            w.requires_grad = name in weights

        opt = torch.optim.Adam(
            list(weights.values()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        for step in range(self.config.num_steps):
            opt.zero_grad()

            if self.config.objective_optimization == "prompt_last":
                inputs = self.tokenizer(
                    [request.prompt], return_tensors="pt", padding=True
                ).to(device)
                target_ids = self.tokenizer(
                    [target], return_tensors="pt", padding=True
                )["input_ids"].to(device)
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                if self.tokenizer.unk_token_id is not None:
                    loss_mask = target_ids != self.tokenizer.unk_token_id
                else:
                    loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
                probs = torch.nn.functional.log_softmax(
                    self.model(**inputs).logits[torch.arange(1), last_token_inds],
                    dim=-1,
                )
                loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(1) / loss_mask.sum(1)
                loss = loss.mean()

            elif self.config.objective_optimization == "target_new":
                full = [request.prompt + target]
                inputs_targets = self.tokenizer(
                    full, return_tensors="pt", padding=True
                ).to(device)
                prompt_inputs = self.tokenizer(
                    [request.prompt], return_tensors="pt", padding=True
                ).to(device)
                num_prompt_toks = int(
                    (prompt_inputs["input_ids"] != self.tokenizer.pad_token_id).sum()
                )
                num_pad_toks = int(
                    (inputs_targets["input_ids"] == self.tokenizer.pad_token_id).sum()
                )
                prompt_len = num_pad_toks + num_prompt_toks
                prompt_target_len = inputs_targets["input_ids"].size(1)
                label_mask = torch.tensor(
                    [[False] * prompt_len + [True] * (prompt_target_len - prompt_len)]
                ).to(device)
                logits = self.model(**inputs_targets).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs_targets["input_ids"][..., 1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss.view(1, -1)
                loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
                loss = loss.mean()
            else:
                raise ValueError(
                    f"Unknown objective: {self.config.objective_optimization}"
                )

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if isinstance(self.config.norm_constraint, float):
                eps = self.config.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v,
                            min=weights_copy[k] - eps,
                            max=weights_copy[k] + eps,
                        )

            if loss.item() < 1e-2:
                break

        # Disable gradient for all params
        for w in self.model.parameters():
            w.requires_grad = False

    def _update_checkpoint(self):
        pass

    def _revert(self):
        pass

