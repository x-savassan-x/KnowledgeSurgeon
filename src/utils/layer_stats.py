from pathlib import Path

import torch
from datasets import load_dataset

from .devices import get_model_device
from .nethook import Trace
from .runningstats import (
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

QWEN_DEFAULT_MAX_LENGTH = 4096


def _advertised_context_length(config) -> int:
    for name in (
        "n_positions",
        "max_sequence_length",
        "max_position_embeddings",
        "seq_length",
    ):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise NotImplementedError("Model config does not expose a context length.")


def _configured_max_length(model, hparams) -> tuple[int, bool]:
    advertised_length = _advertised_context_length(model.config)
    configured_length = getattr(hparams, "layer_stats_max_length", None)
    explicitly_configured = configured_length is not None

    if configured_length is None:
        model_type = str(getattr(model.config, "model_type", "")).casefold()
        if "qwen" in model_type:
            configured_length = QWEN_DEFAULT_MAX_LENGTH

    if configured_length is None:
        return advertised_length, explicitly_configured
    if configured_length <= 0:
        raise ValueError("layer_stats_max_length must be positive when set.")
    return min(advertised_length, int(configured_length)), explicitly_configured


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
    progress=None,
    force_recompute=False,
    hparams=None,
):
    npos, max_length_is_configured = _configured_max_length(model, hparams)
    if batch_tokens is None:
        batch_tokens = npos * 3
    batch_tokens = int(batch_tokens)
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive.")

    def get_ds():
        ds_config_map = {
            "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            "wikipedia": ("wikimedia/wikipedia", "20231101.en"),
        }
        hf_path, hf_config = ds_config_map[ds_name]
        raw_ds = load_dataset(hf_path, hf_config)
        maxlen = min(npos, batch_tokens)
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 100
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    sample_suffix = "" if sample_size is None else f"_{sample_size}"
    length_suffix = f"_l{npos}" if max_length_is_configured else ""
    batch_suffix = f"_t{batch_tokens}" if batch_tokens < npos else ""
    size_suffix = f"{length_suffix}{batch_suffix}{sample_suffix}"
    model_key = getattr(hparams, "model_key", None)
    if model_key is None:
        model_key = model_name
    if model_key is None:
        model_key = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_key}/{ds_name}/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
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
                batch = dict_to_(
                    batch,
                    get_model_device(model, fallback=getattr(hparams, "device", None)),
                )
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
