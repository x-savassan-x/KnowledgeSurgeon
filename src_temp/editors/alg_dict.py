from .ft import apply_ft_to_model
from .rome import apply_rome_to_model
from .memit import apply_memit_to_model
from .grace import apply_grace_to_model
from .wise import apply_wise_to_model
from .alphaedit import apply_AlphaEdit_to_model
from .SPHERE import apply_SPHERE_to_model


ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    'GRACE': apply_grace_to_model,
    'WISE': apply_wise_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
    "SPHERE": apply_SPHERE_to_model,
}