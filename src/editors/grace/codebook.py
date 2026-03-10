import torch


class Codebook(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(Codebook, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.init_epsilon = config.eps_init
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
        keys = torch.vstack(
            [self.keys, new_key.detach()]
        )  # Add new key to list of keys
        values = torch.nn.Parameter(
            torch.vstack([self.values, new_value]), requires_grad=True
        )  # Add new value to list of values
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        if self.epsilons.nelement() == 0:
            epsilons = new_epsilon
        else:
            epsilons = torch.vstack(
                [self.epsilons, new_epsilon]
            )  # Add new epsilon to list of epsilons
        key_labels = [
            self.edit_label
        ] + self.key_labels  # Add new key_label to list of key_labels

        edit_ids = [self.edit_ids] + [new_edit_id]
        return keys, values, epsilons, key_labels, edit_ids

    def delete_key(self, edit_id):
        if "keys" not in self.__dict__ or self.edit_ids == []:
            print("no keys")
            return
        if edit_id in self.edit_ids:
            index_to_remove = self.edit_ids.index(edit_id)
            self.keys = torch.cat(
                (self.keys[:index_to_remove], self.keys[index_to_remove + 1 :]), dim=0
            )
            self.values = torch.nn.Parameter(
                torch.cat(
                    (self.values[:index_to_remove], self.values[index_to_remove + 1 :]),
                    dim=0,
                ),
                requires_grad=True,
            )
            self.epsilons = torch.cat(
                (self.epsilons[:index_to_remove], self.epsilons[index_to_remove + 1 :]),
                dim=0,
            )
            self.key_labels = (
                self.key_labels[:index_to_remove]
                + self.key_labels[index_to_remove + 1 :]
            )
            self.edit_ids = (
                self.edit_ids[:index_to_remove] + self.edit_ids[index_to_remove + 1 :]
            )
            print(
                self.keys.shape,
                self.values.shape,
                self.epsilons.shape,
                len(self.key_labels),
                len(self.edit_ids),
            )
        else:
            print("not found")

    def init_key_value(self, query, value):
        key = query.detach()
        epsilon = torch.tensor(
            self.init_epsilon, device=self.device, requires_grad=False
        ).view(1)
        key_label = [self.edit_label]
        edit_ids = [self.edit_id]
        return key, value, epsilon, key_label, edit_ids

    def label_match(self, edit_label, key_label):
        return edit_label.float().mean() == key_label.float().mean()

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        self.epsilons[nearest_key] = (
            smallest_distance / 2
        ) - 1e-5  # Cut nearest epsilon in half
        self.epsilons[-1] = smallest_distance / 2  # Cut new epsilon in half

    def forward(self, *args):
        # Run layer forward and save what it would have returned for this instance
        layer_out = self.layer(*args)

        ### If training, we need to modify the codebook
        if (not self.training) & ("keys" not in self.__dict__):
            # If it's not training time and we haven't added any keys yet (this is before doing any editing)
            # print(self.__dict__)
            return layer_out
        else:
            if not self.training:
                if self.key_id == -1:
                    token_to_edit = args[0].shape[1] - 1
                    self.key_id = args[0].shape[1] - 1
                else:
                    token_to_edit = min(self.key_id, args[0].shape[1] - 1)
            else:
                token_to_edit = min(
                    self.key_id, args[0].shape[1] - 1
                )  # args[0].shape[1] - 1 is sequence length
            query = args[0][:, token_to_edit, :]  # Just use activation for last token
            if self.config.val_init == "cold":
                new_value = torch.nn.Parameter(
                    torch.rand(
                        1, self.value_shape, requires_grad=True, device=self.device
                    )
                )
            elif self.config.val_init == "warm":
                new_value = torch.nn.Parameter(
                    layer_out[:, token_to_edit, :].detach(), requires_grad=True
                )

            if "keys" not in self.__dict__ or self.keys.nelement() == 0:
                # If no keys exist, initialize keys, values, epsilons, and key labels
                (
                    self.keys,
                    self.values,
                    self.epsilons,
                    self.key_labels,
                    self.edit_ids,
                ) = self.init_key_value(query, new_value)
            elif self.iter == 0:
                # Keys exist, so we have decide whether or not to update them (the fact that we've made it to this point means there was an error!)

                # --- search through keys for a match for query ---
                dists = torch.cdist(self.keys.float(), query.float(), p=2).view(
                    -1, len(query)
                )
                smallest_distance, nearest_key = dists.min(0)

                if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                    # If there's no close key, make a new key
                    (
                        self.keys,
                        self.values,
                        self.epsilons,
                        self.key_labels,
                        self.edit_ids,
                    ) = self.add_key(query, new_value, self.edit_id)
                else:
                    # If there is a close key, we need to handle conflicts
                    if not self.label_match(
                        self.edit_label, self.key_labels[nearest_key]
                    ):
                        (
                            self.keys,
                            self.values,
                            self.epsilons,
                            self.key_labels,
                            self.edit_ids,
                        ) = self.add_key(query, new_value, self.edit_id)
                        self.split_epsilons_in_half(nearest_key, smallest_distance)
                    else:
                        # If the current label is the SAME as the nearest label, just make the nearest epsilon bigger
                        if smallest_distance > self.epsilons[nearest_key]:
                            if self.config.eps_expand == "coverage":
                                self.epsilons[nearest_key] = (
                                    smallest_distance  # Replace nearest epsilon with dist between old key and new key
                                )
                            elif self.config.eps_expand == "moving_average":
                                a = 0.5
                                self.keys[nearest_key] = (
                                    a * self.keys[nearest_key] + (1 - a) * query
                                )  # Move old key to be halfway between
                                self.epsilons[nearest_key] = smallest_distance
                                # self.epsilons[nearest_key] = smallest_distance + self.init_epsilon
            else:
                # If not iter 0, we don't need to change keys, we just need to learn the value
                pass
        # print(token_to_edit)
        # compute distance from query to all keys and find the closest keys

        dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, len(query))
        if dists.nelement() == 0:
            return layer_out
        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        if self.replacement == "replace_all":
            layer_out = torch.where(
                (smallest_dist <= eps).view(-1, 1, 1),
                chosen_value.unsqueeze(1).repeat_interleave(layer_out.shape[1], 1),
                layer_out,
            )
        elif self.replacement == "replace_last":
            layer_out[:, token_to_edit] = torch.where(
                (smallest_dist <= eps), chosen_value, layer_out[:, token_to_edit]
            )
        elif self.replacement == "replace_prompt":
            layer_out[:, :token_to_edit] = torch.where(
                (smallest_dist <= eps), chosen_value, layer_out[:, :token_to_edit]
            )
        else:
            print("token replacement choice not found")
        return layer_out