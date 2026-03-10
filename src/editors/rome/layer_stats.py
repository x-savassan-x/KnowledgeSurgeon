import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.nethook import Trace, set_requires_grad
from ...utils.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    config=None,
):
    """
    Load or compute cached layer statistics (e.g. second moment).
    """

    def get_ds():
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        if hasattr(model.config, "n_positions"):
            maxlen = model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, "max_position_embeddings"):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, "seq_length"):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError(
                "Cannot determine max sequence length from model config"
            )

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 100
    if hasattr(model.config, "n_positions"):
        npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        npos = model.config.seq_length
    else:
        raise NotImplementedError(
            "Cannot determine max sequence length from model config"
        )

    if batch_tokens is None:
        batch_tokens = npos * 3
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = f"_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = (
        f"{model_name}/{ds_name}_stats/"
        f"{layer_name}_{precision}_"
        f"{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    )
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    device = f"cuda:{config.gpu_id}" if config is not None else "cuda"

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with Trace(
                    model,
                    layer_name,
                    retain_input=True,
                    retain_output=False,
                    stop=True,
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat
