from dataclasses import dataclass
from ..config import Config


@dataclass
class GraceConfig(Config):
    # The layer inside the model that GRACE intercepts
    # GRACE replaces this layer with a codebook that checks each incoming activation
    # against stored keys and redirects to the corresponding value vector if it's within
    # a certain distance
    codebook_layer: str

    # Number of gradient steps taken when training each new value vector
    n_iter: int

    # Learning rate for the optimizer used during value training
    edit_lr: float

    # How the value vector is initialized before training
    # 'cold': random initialization
    # 'warm': initialized from the layer's actual output for the edit token
    #         (closer starting point, typically converges faster)
    val_init: str

    # Initial match radius for each codebook entry
    # A query whose distance to a stored key is within eps is redirected
    # to that entry's value vector
    eps_init: float

    # Which token positions in the output get replaced by the stored value
    # when a query matches a codebook key
    #   'replace_last':   only the subject token position
    #   'replace_all':    every token position in the sequence
    #   'replace_prompt': all positions up to (but not including) the subject token
    replacement: str
