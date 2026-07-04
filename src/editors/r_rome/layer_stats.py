import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.devices import get_model_device
from ...utils.nethook import Trace, set_requires_grad
from ...utils.runningstats import (
    CombinedStat,
    Mean,
    NormMean,
    SecondMoment,
    cache_load_enabled,
    tally,
)
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
    progress=None,
    force_recompute=False,
    hparams=None
):
    def get_ds():
        ds_config_map = {
            "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            "wikipedia": ("wikimedia/wikipedia", "20231101.en"),
        }
        hf_path, hf_config = ds_config_map[ds_name]
        raw_ds = load_dataset(hf_path, hf_config)
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, 'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError

        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 100
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, 'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
        npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    tally_kwargs = dict(
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    if force_recompute:
        with cache_load_enabled(False):
            loader = tally(stat, get_ds, **tally_kwargs)
    else:
        loader = tally(stat, get_ds, **tally_kwargs)
    iterator = loader
    if progress is not None and (force_recompute or not filename.exists()):
        batch_count = None
        if sample_size is not None:
            batch_count = -(-int(sample_size) // batch_size)
        iterator = progress(loader, total=batch_count, leave=False)
    with torch.no_grad():
        for batch_group in iterator:
            for batch in batch_group:
                batch = dict_to_(batch, get_model_device(model, fallback=hparams.device))
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(
                    tr.input,
                    batch["attention_mask"].to(tr.input.device),
                )
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat
