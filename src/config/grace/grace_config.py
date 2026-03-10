from dataclasses import dataclass
from ..config import Config


@dataclass
class GraceConfig(Config):
    # The layer inside the model that GRACE intercepts
    # GRACE replaces this layer with an adapter that checks incoming activations
    # against its codebook and returns a stored value when a match is found
    adapter_layer: str

    # How the value vector is initialized before training
    # 'cold': random initialization
    # 'warm': initialized from the layer's actual output for the edit token
    #         (closer starting point, typically converges faster)
    val_init: str

    # Learning rate for the Adam optimizer used during value training
    edit_lr: float

    # Number of gradient steps taken when training each new value vector
    n_iter: int

    # Initial match radius for each codebook entry
    # A query whose distance to a stored key is within eps is redirected
    # to that entry's value vector
    eps_init: float

    # Distance function used to compare a query to stored keys
    #   'euc': Euclidean distance
    #   'mmd': Maximum Mean Discrepancy
    #   'cos': cosine distance
    dist_fn: str

    # Strategy for expanding a key's radius as more edits accumulate
    #   'coverage':   grows to cover nearby queries that should also be redirected
    #   'moving_avg': exponential moving average of observed distances
    #   'decay':      gradually shrinks over time
    eps_expand: str

    # Conflict resolution when a query falls inside an existing key's radius
    # but with a different target label
    #   'replace_last':   replace the most recently added conflicting key
    #   'replace_all':    replace all conflicting keys
    #   'replace_prompt': replace based on prompt identity
    replacement: str
