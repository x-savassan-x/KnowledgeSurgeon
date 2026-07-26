import copy
from typing import Any

import torch

from .utils import brackets_to_periods, parent_module


def perturb_values(
    chosen_value: torch.Tensor,
    num_pert: int,
    device: torch.device | str,
) -> torch.Tensor:
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0] * 0
    noise.requires_grad = True
    chosen_value = chosen_value + noise
    return chosen_value


class GraceCore(torch.nn.Module):
    def __init__(self, config, model, device: torch.device | str) -> None:
        super().__init__()
        self.config = config
        self.log_dict = {}
        self.model = model
        layer = config.inner_params[0]
        self.device = device
        self.original_layer = None

        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for _, p in self.model.named_parameters():
            p.requires_grad = False

        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        self.device = original_layer.weight.device
        if type(original_layer) is not GraceAdapter:
            setattr(edit_module, layer_name, GraceAdapter(config, original_layer).to(self.device))
            self.original_layer = copy.deepcopy(original_layer)

    def __call__(self, **kwargs) -> Any:
        return self.model(**kwargs)

    def reset_layer(self) -> None:
        layer_name = self.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        if self.original_layer is not None:
            setattr(edit_module, layer_name, self.original_layer.to(self.device))

    def set_inference_mode(self) -> None:
        adapter = eval(f"self.model.{self.layer}")
        setattr(adapter, "training", False)
        setattr(adapter, "key_id", -1)
        setattr(adapter, "iter", -1)

    def generate(self, *args, **kwargs) -> Any:
        self.set_inference_mode()
        return self.model.generate(*args, **kwargs)

    def rolllback(self, edit_id: int) -> None:
        layer = eval(f"self.model.{self.layer}")
        layer.delete_key(edit_id)

    def edit(self, config, tokens: dict[str, torch.Tensor], edit_id: int) -> None:
        key_id = (tokens["labels"] == -100).sum() - 1
        adapter = eval(f"self.model.{self.layer}")
        setattr(adapter, "key_id", key_id)
        setattr(adapter, "training", True)
        setattr(adapter, "edit_label", tokens["labels"])
        setattr(adapter, "edit_id", edit_id)
        adapter.reset_training_state()

        self.losses = []
        optimizer = None
        loss = None
        try:
            for i in range(config.n_iter):
                adapter.iter = i
                outputs = self.model(**tokens)
                if i == 0:
                    optimizer = torch.optim.Adam([adapter.active_value], config.edit_lr, foreach=False)
                loss = outputs.loss
                loss.backward()
                if optimizer is None:
                    raise RuntimeError("GRACE optimizer was not initialized")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.losses.append(loss.detach().cpu().item())

            if loss is None:
                raise RuntimeError("GRACE edit did not run any optimization step")
            adapter.commit_trained_value()
        except BaseException:
            adapter.reset_training_state()
            adapter.training = False
            raise

        self.loss = loss.detach().cpu()
        adapter.training = False
        chosen_key = adapter.chosen_key.detach().cpu()
        nkeys = len(adapter.keys)

        self.log_dict["chosen_key"] = chosen_key
        self.log_dict["nkeys"] = nkeys


class GraceAdapter(torch.nn.Module):
    CAPACITY_BLOCK = 512

    def __init__(self, config, layer: torch.nn.Module) -> None:
        super().__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.init_epsilon = config.eps
        self.dist_fn = config.dist_fn
        self.replacement = config.replacement
        self.device = layer.weight.device
        self.config = config
        self.num_pert = config.num_pert
        self.key_id = -1
        self.iter = -1
        self.ensure_replace_token_loc = False

        self.key_shape = layer.weight.shape[1]
        self.value_shape = layer.weight.shape[0]
        self.training = False

        value_dtype = torch.float32 if config.val_init == "cold" else layer.weight.dtype
        self.register_buffer(
            "_key_storage",
            torch.empty(0, self.key_shape, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_value_storage",
            torch.empty(0, self.value_shape, dtype=value_dtype, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_epsilon_storage",
            torch.empty(0, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self._num_keys = 0
        self.key_labels = []
        self.edit_ids = []
        self._training_smallest_dist = None
        self._training_chosen_key = None

    @property
    def keys(self) -> torch.Tensor:
        return self._key_storage[:self._num_keys]

    @property
    def values(self) -> torch.Tensor:
        return self._value_storage[:self._num_keys]

    @property
    def epsilons(self) -> torch.Tensor:
        return self._epsilon_storage[:self._num_keys]

    def ensure_capacity(self, required: int) -> None:
        current = len(self._key_storage)
        if required <= current:
            return

        capacity = ((required + self.CAPACITY_BLOCK - 1) // self.CAPACITY_BLOCK) * self.CAPACITY_BLOCK
        key_storage = torch.empty(
            capacity,
            self.key_shape,
            dtype=self._key_storage.dtype,
            device=self.device,
        )
        value_storage = torch.empty(
            capacity,
            self.value_shape,
            dtype=self._value_storage.dtype,
            device=self.device,
        )
        epsilon_storage = torch.empty(
            capacity,
            dtype=self._epsilon_storage.dtype,
            device=self.device,
        )
        if self._num_keys:
            key_storage[:self._num_keys].copy_(self.keys)
            value_storage[:self._num_keys].copy_(self.values)
            epsilon_storage[:self._num_keys].copy_(self.epsilons)
        self._key_storage = key_storage
        self._value_storage = value_storage
        self._epsilon_storage = epsilon_storage

    def load_codebook(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        epsilons: torch.Tensor,
    ) -> None:
        num_keys = len(keys)
        self.ensure_capacity(num_keys)
        if num_keys:
            self._key_storage[:num_keys].copy_(keys)
            self._value_storage[:num_keys].copy_(values)
            self._epsilon_storage[:num_keys].copy_(epsilons.view(-1))
        self._num_keys = num_keys

    def add_key(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        new_edit_id: int,
    ) -> None:
        self.ensure_capacity(self._num_keys + 1)
        self._key_storage[self._num_keys].copy_(new_key.detach().float().view(-1))
        self._value_storage[self._num_keys].copy_(new_value.detach().view(-1))
        self._epsilon_storage[self._num_keys] = self.init_epsilon
        self._num_keys += 1
        # Keep metadata in the same row order as keys, values, and epsilons.
        self.key_labels = self.key_labels + [self.edit_label]
        self.edit_ids.append(new_edit_id)

    def delete_key(self, edit_id: int) -> None:
        if not self.edit_ids:
            return
        if edit_id in self.edit_ids:
            index_to_remove = self.edit_ids.index(edit_id)
            num_to_move = self._num_keys - index_to_remove - 1
            if num_to_move:
                source = slice(index_to_remove + 1, self._num_keys)
                target = slice(index_to_remove, self._num_keys - 1)
                self._key_storage[target].copy_(self._key_storage[source].clone())
                self._value_storage[target].copy_(self._value_storage[source].clone())
                self._epsilon_storage[target].copy_(self._epsilon_storage[source].clone())
            self._num_keys -= 1
            self.key_labels = self.key_labels[:index_to_remove] + self.key_labels[index_to_remove + 1 :]
            self.edit_ids = self.edit_ids[:index_to_remove] + self.edit_ids[index_to_remove + 1 :]

    def init_key_value(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.add_key(query, value, self.edit_id)

    def reset_training_state(self) -> None:
        if hasattr(self, "active_value"):
            del self.active_value
        self._training_smallest_dist = None
        self._training_chosen_key = None

    def start_value_training(self, chosen_key: torch.Tensor) -> None:
        self._training_chosen_key = chosen_key.detach().clone()
        self.active_value = torch.nn.Parameter(
            self.values[self._training_chosen_key].detach().clone(),
            requires_grad=True,
        )

    def commit_trained_value(self) -> None:
        if self._training_chosen_key is None or not hasattr(self, "active_value"):
            raise RuntimeError("GRACE has no trained value to commit")
        with torch.no_grad():
            self.values[self._training_chosen_key] = self.active_value.detach()
        del self.active_value
        self._training_smallest_dist = None
        self._training_chosen_key = None

    def initialize_value(self, layer_out: torch.Tensor, token_to_edit: int) -> torch.Tensor:
        if self.config.val_init == "cold":
            return torch.rand(1, self.value_shape, device=self.device)
        if self.config.val_init == "warm":
            return layer_out[:, token_to_edit, :].detach()
        raise ValueError(f"Unsupported val_init: {self.config.val_init}")

    def distances(self, query: torch.Tensor) -> torch.Tensor:
        return torch.cdist(self.keys, query.float(), p=2).view(-1, len(query))

    def label_match(self, edit_label: torch.Tensor, key_label: torch.Tensor) -> torch.Tensor:
        return edit_label.float().mean() == key_label.float().mean()

    def split_epsilons_in_half(
        self,
        nearest_key: torch.Tensor,
        smallest_distance: torch.Tensor,
    ) -> None:
        self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5
        self.epsilons[-1] = smallest_distance / 2

    def forward(self, *args) -> Any:
        layer_out = self.layer(*args)

        if not self.training and self._num_keys == 0:
            return layer_out
        else:
            if not self.training:
                if self.key_id == -1:
                    token_to_edit = args[0].shape[1] - 1
                    self.key_id = args[0].shape[1] - 1
                else:
                    token_to_edit = min(self.key_id, args[0].shape[1] - 1)
            else:
                token_to_edit = min(self.key_id, args[0].shape[1] - 1)
            query = args[0][:, token_to_edit, :]
            dists = None

            if self._num_keys == 0:
                new_value = self.initialize_value(layer_out, token_to_edit)
                self.init_key_value(query, new_value)
            elif self.training and self.iter == 0:
                dists = self.distances(query)
                smallest_distance, nearest_key = dists.min(0)

                if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                    new_value = self.initialize_value(layer_out, token_to_edit)
                    self.add_key(query, new_value, self.edit_id)
                    dists = None
                else:
                    if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                        new_value = self.initialize_value(layer_out, token_to_edit)
                        self.add_key(query, new_value, self.edit_id)
                        self.split_epsilons_in_half(nearest_key, smallest_distance)
                        dists = None
                    else:
                        if smallest_distance > self.epsilons[nearest_key]:
                            if self.config.eps_expand == "coverage":
                                self.epsilons[nearest_key] = smallest_distance
                            elif self.config.eps_expand == "moving_average":
                                a = 0.5
                                self.keys[nearest_key] = a * self.keys[nearest_key] + (1 - a) * query
                                self.epsilons[nearest_key] = smallest_distance
                                dists = None

        if self.training and self.iter > 0:
            if self._training_smallest_dist is None or self._training_chosen_key is None:
                raise RuntimeError("GRACE training selection was not initialized")
            smallest_dist = self._training_smallest_dist
            self.chosen_key = self._training_chosen_key
        else:
            if dists is None:
                dists = self.distances(query)
            if dists.nelement() == 0:
                return layer_out
            smallest_dist, self.chosen_key = dists.min(0)
            if self.training:
                self._training_smallest_dist = smallest_dist.detach()
                self.start_value_training(self.chosen_key)

        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.active_value if self.training else self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        if self.config.val_train in {"adv", "pert"} and self.training:
            chosen_value = perturb_values(chosen_value, self.num_pert, self.device)

        # During training we must always route the edited token through the trainable
        # value tensor; otherwise the model can fall back to the frozen base layer and
        # the resulting loss becomes detached from autograd.
        should_replace = torch.ones_like(smallest_dist, dtype=torch.bool) if self.training else (smallest_dist <= eps)

        if self.replacement == "replace_all":
            layer_out = torch.where(
                should_replace.view(-1, 1, 1),
                chosen_value.unsqueeze(1).repeat_interleave(layer_out.shape[1], 1),
                layer_out,
            )
        elif self.replacement == "replace_last":
            layer_out[:, token_to_edit] = torch.where(should_replace, chosen_value, layer_out[:, token_to_edit])
        elif self.replacement == "replace_prompt":
            layer_out[:, :token_to_edit] = torch.where(should_replace, chosen_value, layer_out[:, :token_to_edit])

        return layer_out
