import copy

import torch
import transformers

from .utils import brackets_to_periods, parent_module


def perturb_values(chosen_value, num_pert, device):
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0] * 0
    noise.requires_grad = True
    chosen_value = chosen_value + noise
    return chosen_value


class GraceCore(torch.nn.Module):
    def __init__(self, config, model, device):
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

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        if type(original_layer) is not GraceAdapter:
            setattr(edit_module, layer_name, GraceAdapter(config, original_layer, transpose=transpose).to(self.device))
            self.original_layer = copy.deepcopy(original_layer)

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def reset_layer(self):
        layer_name = self.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        if self.original_layer is not None:
            setattr(edit_module, layer_name, self.original_layer.to(self.device))

    def set_inference_mode(self):
        adapter = eval(f"self.model.{self.layer}")
        setattr(adapter, "training", False)
        setattr(adapter, "key_id", -1)

    def generate(self, *args, **kwargs):
        self.set_inference_mode()
        return self.model.generate(*args, **kwargs)

    def rolllback(self, edit_id):
        layer = eval(f"self.model.{self.layer}")
        layer.delete_key(edit_id)

    def edit(self, config, tokens, edit_id):
        key_id = (tokens["labels"] == -100).sum() - 1
        adapter = eval(f"self.model.{self.layer}")
        setattr(adapter, "key_id", key_id)

        # __init__ blanket-disables requires_grad on every model parameter,
        # including the adapter values from a previous edit.  Re-enable so
        # the optimiser can still back-prop into them.
        if hasattr(adapter, "values") and isinstance(adapter.values, torch.nn.Parameter):
            adapter.values.requires_grad_(True)

        setattr(adapter, "training", True)
        setattr(adapter, "edit_label", tokens["labels"])
        setattr(adapter, "edit_id", edit_id)

        self.losses = []
        optimizer = None
        loss = None
        for i in range(config.n_iter):
            setattr(eval(f"self.model.{self.layer}"), "iter", i)

            outputs = self.model(**tokens)
            if i == 0:
                optimizer = torch.optim.Adam(self.model.parameters(), config.edit_lr)
            loss = outputs.loss
            loss.backward()
            if optimizer is None:
                raise RuntimeError("GRACE optimizer was not initialized")
            optimizer.step()
            optimizer.zero_grad()
            self.losses.append(loss.detach().cpu().numpy())

        if loss is None:
            raise RuntimeError("GRACE edit did not run any optimization step")
        self.loss = loss

        setattr(eval(f"self.model.{self.layer}"), "training", False)
        chosen_key = getattr(eval(f"self.model.{self.layer}"), "chosen_key")
        nkeys = len(getattr(eval(f"self.model.{self.layer}"), "keys"))

        self.log_dict["chosen_key"] = chosen_key
        self.log_dict["nkeys"] = nkeys


class GraceAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
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
        self.ensure_replace_token_loc = False

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False

    def add_key(self, new_key, new_value, new_edit_id):
        keys = torch.vstack([self.keys, new_key.detach()])
        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True)
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        if self.epsilons.nelement() == 0:
            epsilons = new_epsilon
        else:
            epsilons = torch.vstack([self.epsilons, new_epsilon])
        key_labels = [self.edit_label] + self.key_labels

        edit_ids = self.edit_ids + [new_edit_id]
        return keys, values, epsilons, key_labels, edit_ids

    def delete_key(self, edit_id):
        if "keys" not in self.__dict__ or self.edit_ids == []:
            return
        if edit_id in self.edit_ids:
            index_to_remove = self.edit_ids.index(edit_id)
            self.keys = torch.cat((self.keys[:index_to_remove], self.keys[index_to_remove + 1 :]), dim=0)
            self.values = torch.nn.Parameter(
                torch.cat((self.values[:index_to_remove], self.values[index_to_remove + 1 :]), dim=0),
                requires_grad=True,
            )
            self.epsilons = torch.cat((self.epsilons[:index_to_remove], self.epsilons[index_to_remove + 1 :]), dim=0)
            self.key_labels = self.key_labels[:index_to_remove] + self.key_labels[index_to_remove + 1 :]
            self.edit_ids = self.edit_ids[:index_to_remove] + self.edit_ids[index_to_remove + 1 :]

    def init_key_value(self, query, value):
        key = query.detach()
        epsilon = torch.tensor(self.init_epsilon, device=self.device, requires_grad=False).view(1)
        key_label = [self.edit_label]
        edit_ids = [self.edit_id]
        return key, value, epsilon, key_label, edit_ids

    def label_match(self, edit_label, key_label):
        return edit_label.float().mean() == key_label.float().mean()

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5
        self.epsilons[-1] = smallest_distance / 2

    def forward(self, *args):
        layer_out = self.layer(*args)

        if not self.training and "keys" not in self.__dict__:
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
            new_value = None
            if self.config.val_init == "cold":
                new_value = torch.nn.Parameter(torch.rand(1, self.value_shape, requires_grad=True, device=self.device))
            elif self.config.val_init == "warm":
                new_value = torch.nn.Parameter(layer_out[:, token_to_edit, :].detach(), requires_grad=True)
            else:
                raise ValueError(f"Unsupported val_init: {self.config.val_init}")

            if "keys" not in self.__dict__ or self.keys.nelement() == 0:
                self.keys, self.values, self.epsilons, self.key_labels, self.edit_ids = self.init_key_value(query, new_value)
            elif self.iter == 0:
                dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, len(query))
                smallest_distance, nearest_key = dists.min(0)

                if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                    self.keys, self.values, self.epsilons, self.key_labels, self.edit_ids = self.add_key(query, new_value, self.edit_id)
                else:
                    if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                        self.keys, self.values, self.epsilons, self.key_labels, self.edit_ids = self.add_key(query, new_value, self.edit_id)
                        self.split_epsilons_in_half(nearest_key, smallest_distance)
                    else:
                        if smallest_distance > self.epsilons[nearest_key]:
                            if self.config.eps_expand == "coverage":
                                self.epsilons[nearest_key] = smallest_distance
                            elif self.config.eps_expand == "moving_average":
                                a = 0.5
                                self.keys[nearest_key] = a * self.keys[nearest_key] + (1 - a) * query
                                self.epsilons[nearest_key] = smallest_distance

        dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, len(query))
        if dists.nelement() == 0:
            return layer_out
        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
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
