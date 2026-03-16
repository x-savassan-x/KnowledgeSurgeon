import torch

from .codebook import Codebook
from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import GraceConfig


class GRACE(Editor):
    config: GraceConfig

    def __init__(self, config: GraceConfig):
        super().__init__(config)
        self.tokenizer.padding_side = 'left'

        for p in self.model.parameters():
            p.requires_grad = False

        orig_module = self.model.get_submodule(config.codebook_layer)
        self.codebook = Codebook(orig_module, config).to(self.model.device)
        self.model.set_submodule(config.codebook_layer, self.codebook, strict=True)

        self._checkpoint = None

    def reset(self):
        self.codebook.keys = None
        self.codebook.values = None
        self.codebook.epsilons = None
        self.codebook.key_labels = None
        self._checkpoint = None

    def _apply(self, request: EditRequest):
        prompt, ans = request.prompt, request.new_ans
        tokens = self.tokenizer(f"{prompt} {ans}", return_tensors="pt")
        tokens["labels"] = tokens["input_ids"].clone()
        prompt_tokens_len = len(self.tokenizer.encode(prompt))
        tokens["labels"][0, :prompt_tokens_len] = -100 # ignore_index of CrossEntropyLoss
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        self.codebook.edit_label = tokens["labels"]

        self.codebook.training = True
        self.codebook.key_id = (tokens["labels"] == -100).sum() - 1

        for i in range(self.config.n_iter):
            self.codebook.iter = i
            outputs = self.model(**tokens)
            if i == 0:
                optimizer = torch.optim.Adam(self.model.parameters(), self.config.edit_lr)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.codebook.training = False
        self.codebook.iter = -1
        self.codebook.key_id = -1

    def _update_checkpoint(self):
        self._checkpoint = {
            "keys":       self.codebook.keys.detach().clone(),
            "values":     self.codebook.values.detach().clone(),
            "epsilons":   self.codebook.epsilons.detach().clone(),
            "key_labels": list(self.codebook.key_labels),
        }

    def _revert(self):
        if self._checkpoint is None:
            self.reset()
        else:
            self.codebook.keys       = self._checkpoint["keys"]
            self.codebook.values     = torch.nn.Parameter(self._checkpoint["values"].clone())
            self.codebook.epsilons   = self._checkpoint["epsilons"]
            self.codebook.key_labels = list(self._checkpoint["key_labels"])
