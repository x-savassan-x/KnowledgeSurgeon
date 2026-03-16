import torch
from ...config import GraceConfig


class Codebook(torch.nn.Module):
    def __init__(self, orig_module: torch.nn.Linear, config: GraceConfig):
        super(Codebook, self).__init__()

        self.orig_module = orig_module
        self.config = config

        self.keys: torch.Tensor | None = None
        self.values: torch.nn.Parameter | None = None
        self.epsilons: torch.Tensor | None = None
        self.key_labels: list[torch.Tensor] | None = None

        self.edit_label = torch.Tensor()
        self.training = False
        self.key_id = -1
        self.iter = -1

        self.key_shape = orig_module.weight.shape[1]
        self.value_shape = orig_module.weight.shape[0]
        orig_module.weight.T

        self.device = orig_module.weight.device

    def forward(self, input):
        output = self.orig_module(input)

        if (not self.training) and (self.keys is None):
            return output

        token_to_edit = min(self.key_id, input.shape[1] - 1)
        query = input[:, token_to_edit, :]

        match self.config.val_init:
            case "cold":
                new_value = torch.nn.Parameter(torch.rand(1, self.value_shape, device=self.device))
            case "warm":
                new_value = torch.nn.Parameter(output[:, token_to_edit, :].detach())

        if self.keys is None:
            self.init_key_value(query, new_value)

        elif self.iter == 0:
            assert self.keys is not None\
                and self.values is not None\
                and self.epsilons is not None\
                and self.edit_label is not None\
                and self.key_labels is not None

            dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, len(query))
            smallest_distance, nearest_key = dists.min(0)

            if smallest_distance > self.config.eps_init + self.epsilons[nearest_key]:
                self.add_key(query, new_value)
            else:
                if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                    self.add_key(query, new_value)
                    self.split_epsilons_in_half(nearest_key, smallest_distance)
                elif smallest_distance > self.epsilons[nearest_key]:
                    self.epsilons[nearest_key] = smallest_distance

        dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, len(query))
        smallest_dist, chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[chosen_key]
        eps = self.epsilons[chosen_key].view(-1, 1)

        match self.config.replace_strat:
            case "replace_last":
                output[:, token_to_edit, :] = torch.where(
                    (smallest_dist <= eps),
                    chosen_value,
                    output[:, token_to_edit, :]
                )
            case "replace_prompt":
                output[:, :token_to_edit, :] = torch.where(
                    (smallest_dist <= eps),
                    chosen_value,
                    output[:, :token_to_edit, :]
                )
            case "replace_all":
                output = torch.where(
                    (smallest_dist <= eps).view(-1, 1, 1),
                    chosen_value.unsqueeze(1).repeat_interleave(output.shape[1], 1),
                    output,
                )

        return output

    def init_key_value(self, query: torch.Tensor, value: torch.nn.Parameter):
        self.keys = query.detach()
        self.values = value
        self.epsilons = torch.tensor(self.config.eps_init, device=self.device).view(1)
        self.key_labels = [self.edit_label]

    def add_key(self, new_key: torch.Tensor, new_value: torch.nn.Parameter):
        self.keys = torch.vstack([self.keys, new_key.detach()])

        self.values = torch.nn.Parameter(torch.vstack([self.values, new_value]))

        new_epsilon = torch.tensor(self.config.eps_init, device=self.device).view(1)
        self.epsilons = torch.vstack([self.epsilons, new_epsilon])

        self.key_labels.append(self.edit_label)

    def label_match(self, edit_label: torch.Tensor, key_label: torch.Tensor):
        return edit_label.float().mean() == key_label.float().mean()

    def split_epsilons_in_half(self, nearest_key: torch.Tensor, smallest_distance: torch.Tensor):
        self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5
        self.epsilons[-1] = smallest_distance / 2
