"""
Evaluation utilities for model editing methods.
Computes accuracy, generalization, locality, and portability metrics.
"""

import typing
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from ..utils import HyperParams
from .evaluate_utils import (
    test_prediction_acc,
    test_prediction_acc_LLM_judge,
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the dataset record). Returns a dictionary
    containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: Dataset record with prompt, target_new, ground_truth, etc.
    :return: Dictionary containing rewriting metrics
    """
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    key = 'rephrase' if test_rephrase else 'rewrite'

    # LLM-as-a-Judge evaluation
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        acc, gen_content = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_acc": acc,
            f"{key}_gen_content": gen_content
        }
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "generate-text":
        gen_content_model = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_gen_content": gen_content_model
        }
    else:
        # Traditional teacher-forcing evaluation (GRACE uses vanilla generation)
        acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device,
                                  vanilla_generation=(hparams.alg_name == "GRACE"))
        ret = {
            f"{key}_acc": acc
        }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if hasattr(hparams, 'evaluation_type'):
        loc_tokens = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:
        loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device,
                                         locality=True, vanilla_generation=(hparams.alg_name == 'GRACE'))
        if type(loc_tokens) is not list:
            loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type in ("LLM-judge", "generate-text"):
        portability_correct = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, ground_truth, device, locality=False)
    else:
        portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device,
                                                  vanilla_generation=(hparams.alg_name == 'GRACE'))

    ret = {
        f"{portability_key}_acc": portability_correct
    }
    return ret
