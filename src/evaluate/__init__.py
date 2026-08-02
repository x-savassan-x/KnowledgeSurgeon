from importlib import import_module
from typing import Any

from .schema import (
    EvalRecord,
    EvalResult,
    MetricResult,
)
from .io import (
    EVALUATION_DIR,
    build_experiment_dir,
    build_hardware_path,
    build_method_dir,
    build_milestone_result_path,
    build_result_path,
    iter_result_files,
    load_hardware_results,
    load_result_file,
    save_benchmark_results,
    save_hardware_results,
    save_memory_results,
    save_scored_results,
    save_time_results,
    save_unscored_results,
)


_LAZY_EXPORTS = {
    "AsyncGenerationManager": ".generation",
    "async_generation_disabled_reason": ".generation",
    "async_generation_device_indices": ".generation",
    "generate_and_save_unscored": ".generation",
    "generate_predictions": ".generation",
    "get_model_predictions": ".generation",
    "model_device_indices": ".generation",
    "collect_hardware_info": ".hardware",
    "find_scored_files": ".judge_utils",
    "find_unscored_files": ".judge_utils",
    "get_judge_scores": ".judge",
    "materialize_base_perf_results": ".judge",
    "rebuild_perf_for_scored_paths": ".judge",
    "rebuild_perf_from_scored": ".judge",
    "rebuild_perf_results": ".judge",
    "score_eval_results": ".judge",
    "score_pending_unscored": ".judge",
    "score_serialized_eval_result": ".judge",
    "score_unscored_file": ".judge",
    "save_run_hardware": ".hardware",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "AsyncGenerationManager",
    "EVALUATION_DIR",
    "EvalRecord",
    "EvalResult",
    "MetricResult",
    "async_generation_disabled_reason",
    "async_generation_device_indices",
    "build_experiment_dir",
    "build_hardware_path",
    "build_method_dir",
    "build_milestone_result_path",
    "build_result_path",
    "collect_hardware_info",
    "find_scored_files",
    "find_unscored_files",
    "generate_and_save_unscored",
    "generate_predictions",
    "get_judge_scores",
    "get_model_predictions",
    "iter_result_files",
    "load_hardware_results",
    "load_result_file",
    "materialize_base_perf_results",
    "model_device_indices",
    "rebuild_perf_for_scored_paths",
    "rebuild_perf_from_scored",
    "rebuild_perf_results",
    "save_benchmark_results",
    "save_hardware_results",
    "save_memory_results",
    "save_run_hardware",
    "save_scored_results",
    "save_time_results",
    "save_unscored_results",
    "score_eval_results",
    "score_pending_unscored",
    "score_serialized_eval_result",
    "score_unscored_file",
]
