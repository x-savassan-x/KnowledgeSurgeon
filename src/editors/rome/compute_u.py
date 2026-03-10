from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from .layer_stats import layer_stats
from ...config.rome import RomeConfig

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    config: RomeConfig,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for "
            f"{model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            config.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            config=config,
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to(f"cuda:{config.gpu_id}")
        ).float()

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    config: RomeConfig,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the left vector used in constructing the rank-1 update matrix.
    """
    print("Computing left vector (u)...")

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=config.rewrite_module_tmp,
        track="in",
    )

    if "subject_" in config.fact_token and config.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")

        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=config.fact_token[len("subject_"):],
            **word_repr_args,
        ).mean(0)
    elif config.fact_token == "last":
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={config.fact_token} not recognized")

    u = cur_repr
    if config.mom2_adjustment:
        inv_cov = get_inv_cov(
            model,
            tok,
            config.rewrite_module_tmp.format(layer),
            config.mom2_dataset,
            config.mom2_n_samples,
            config.mom2_dtype,
            config=config,
        ).to(device=u.device, dtype=u.dtype)
        u = inv_cov @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
