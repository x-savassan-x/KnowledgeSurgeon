# Evaluation Pipeline Context

## Current Dataset State

- Final milestones are `10, 100, 1000, 5000, 10000`, with subsets such as `10, 100` used for quick testing.
- Dataset variants are being removed. The pipeline should no longer assume separate `cloze` and `qna` configs, paths, or HuggingFace splits.
- `es_loc` local source is currently derived from CounterFact at `data/datasets/counterfact/enhanced`; evaluation loads the published HuggingFace artifact directly from config `es_loc`.
- `es_loc` current schema has 17,125 records:
  - `edit.prompt`
  - `edit.subject`
  - `edit.answer.true`
  - `edit.answer.new`
  - `generality`: list of prompt strings, all judged against `edit.answer.new`
  - `locality`: list of `{prompt, answer}` objects
- `port` local source is currently based on MQuAKE and will later be merged/expanded with UniEdit as needed; final portability artifact should be loadable from HuggingFace config `port` once ready.
- `port` current MQuAKE-style schema has 7,550 records:
  - `edit.prompt`
  - `edit.subject.{id,str}`
  - `edit.relation.{id,str}`
  - `edit.answer.true.{id,str}`
  - `edit.answer.new.{id,str}`
  - `portability.prompts`: list of three questions
  - `portability.answers.true`: accepted true-answer strings
  - `portability.answers.new`: accepted new-answer strings
  - `portability.hops.true` and `portability.hops.new`: two-hop metadata with prompt, subject, relation, and accepted answer strings
  - `portability.rewrite_required`: temporary preprocessing marker, should not exist in final rewritten artifacts.
- MQuAKE rewrite code in `notebook/data.ipynb` uses a canonical two-hop question and rewrites only records that need prompt repair. The current workspace does not have a saved `data/datasets/mquake-remastered/rewritten` artifact.
- UniEdit remains downloaded at `data/datasets/uniedit/raw`, but it is not finished. Even if UniEdit is used later, portability data should follow the latest MQuAKE uniform schema. MQuAKE rewrite only changes prompt content, not schema.

## Current Evaluation Layout

- Dataset variants have been removed from the active evaluation path.
- Shared producer/worker settings live in `src/evaluate/config.yaml`. Both `notebook/edit.ipynb` and `notebook/eval.ipynb` load this file, so method lists, result names, milestones, run counts, judge repeats, and concurrency settings should be changed there first.
- `notebook/edit.ipynb` is the active producer notebook. It loads `es_loc` directly from HuggingFace, runs editing/model generation, and writes `unscored/`, `time/`, and `mem/`.
- `notebook/eval.ipynb` is the active judge-worker notebook. It watches pending `unscored/` files, writes `scored/`, and rebuilds `perf/`.
- `notebook/evaluate.ipynb` is the old combined notebook and should be treated as legacy while the split workflow is used.
- Portability loading is already written to use HuggingFace config `port`, but remains commented until the final artifact is ready.
- Result IO and visualization use the no-variant layout:
  - `data/evaluation/{model}/{milestones}/r{n_runs}-j{judge_repeats}/{method}/`
  - behavioral result files live under each method directory's `perf` subfolder, e.g. `perf/es_loc.*.json` and `perf/port.*.json`
  - time-related information should not be duplicated inside behavioral result files
  - generated-but-unscored milestone snapshots live under `unscored`, e.g. `unscored/es_loc.{run}.{step}.json`
  - time data should live under each method directory's `time` subfolder, e.g. `time/es_loc.*.json` and `time/port.*.json`
  - time files should contain only `edit`, a raw per-step edit-time list.
  - summaries such as total edit time and average edit time at each milestone should be computed on demand from the raw edit-time list.

## Required Schema Mapping

- `es_loc` should build one edit request from:
  - subject: `record["edit"]["subject"]`
  - prompt: `record["edit"]["prompt"]`
  - old answer: `record["edit"]["answer"]["true"]`
  - new answer: `record["edit"]["answer"]["new"]`
- `es_loc` success records should include the edit prompt plus all `generality` prompts, scored against `edit.answer.new`.
- `es_loc` locality records should use each `locality` object, scored against its own `answer`.
- `port` should build one edit request from:
  - subject string: `record["edit"]["subject"]["str"]`
  - prompt: `record["edit"]["prompt"]`
  - old answer: `record["edit"]["answer"]["true"]["str"]`
  - new answer: `record["edit"]["answer"]["new"]["str"]`
- `port` portability records should use `portability.prompts`, scored against `portability.answers.new`.
- Evaluation should be alias-aware through `EvalRecord.ground_truths: list[str]`; single-answer records should pass a one-element list. Judge prompts and result serialization must reflect this change.

## Benchmark Execution Direction

- Current execution is synchronous at milestones: edit until a milestone, stop, evaluate that model state, then continue editing.
- Preferred next target is asynchronous judge scoring without checkpointing:
  - at a milestone, run model generation for the current eval records while the model is still at that milestone state,
  - save or enqueue the generated-but-unscored records,
  - continue the edit stream immediately,
  - let judge scoring consume pending unscored records in the background.
- This is preferred because model generation depends on model state but is comparatively batchable, while judge scoring is the long independent stage.
- Current implementation uses a process-level producer/consumer boundary:
  - milestone generation saves unscored snapshots immediately,
  - editing continues after each unscored snapshot is written,
  - the editing notebook does not score snapshots or write final `perf` files,
  - a separate judge worker process consumes `unscored/`, writes `scored/`, and rebuilds `perf/`.
- The separate worker should be run in another Python process, notebook kernel, or machine. This avoids the event-loop limitation where judge tasks only progress between blocking edit/generation calls.
- Worker-level file concurrency is controlled by `score_pending_unscored(..., file_concurrency=...)`; judge request concurrency is still controlled by `judge_config["max_concurrency"]`.
- Checkpoint/state saving is deprioritized for now. It remains an option only if generation itself becomes a bottleneck or if full asynchronous milestone evaluation is needed later.
- Runs and methods are independent and can be parallelized at the scheduler level when enough machines are available.
- Judge repetitions can also be parallelized because each repeat is independent, but this mainly shifts load to the judge server and should be controlled by judge concurrency.

## Time And Composite Direction

- Milestone-specific `edit_time` is not ideal for Mnemo because milestones commonly align with consolidation cycles, making those points look artificially slow.
- Sequential time visualization should use average edit time so far at each milestone: `total_edit_time / step`.
- Milestone composite should use milestone performance scores and average edit time so far at that milestone.
- Final bar choice is still open:
  - final-milestone value is better for stress-test end-state comparison,
  - mean-over-milestones is better for whole-trajectory comparison,
  - both may be worth reporting separately if space allows.
- Composite should remain derived from performance and time values, not from averaging already-computed composite values unless explicitly labeled as trajectory mean.

## Side Telemetry To Add

- Keep telemetry separate from behavioral result files and raw edit-time arrays.
- Ignore one-time setup/precompute telemetry for now, e.g. MEMIT layer statistics or AlphaEdit projection calculation.
- Runtime timing stored in `time/{result_name}.{run}.json` should contain only:
  - `edit`: full per-step edit-time list.
- Runtime memory should be stored separately under `mem/{result_name}.{run}.json` only for edit-time memory usage at first.
- Start memory telemetry with peak CUDA VRAM during editing because edit memory is the most deterministic method-level resource signal. Host RAM can be added later if needed. Avoid high-frequency memory traces unless peak values are insufficient.
- Avoid treating eval time, judge time, eval memory, judge memory, or whole-orchestrator pipeline memory/time as core metrics because those values depend heavily on scheduling, machine count, device choice, and parallelism strategy.
- Default full-experiment setting is `r3-j3`: three independent edit/data runs and three judge repeats.
- Runs and methods are independent and can be parallelized when enough GPUs/memory are available. Telemetry should therefore be collected at method/run/event level for consistency; cluster-level pipeline telemetry is optional and should be reported separately if used.

## Method Assets

- Benchmark datasets are loaded directly from the HuggingFace dataset repo in `notebook/edit.ipynb`; dataset upload/reconstruction remains in `notebook/data.ipynb`.
- `notebook/method_assets.ipynb` is only for precomputed method assets needed by editing methods:
  - MEMIT/ROME-style layer statistics under `data/layer_stats/**/*.npz`.
  - AlphaEdit null-space projection under `data/alpha_edit_projectors/null_space_project.pt`.
- The notebook keeps the existing remote base `artifacts/qwen2.5-7b` because these assets are model-specific, unlike the shared evaluation run config in `src/evaluate/config.yaml`.

## Hardware Planning Context

- Intended rental topology is larger than a single edit/eval pair:
  - three edit/eval GPU pairs, likely two pairs of 24GB GPUs and one pair of 32GB GPUs
  - each pair runs one editing process and one evaluation process
  - a separate judge-hosting GPU may be rented to maximize judge concurrency
  - each edit/eval machine is expected to have around 150GB host memory, subject to checkpoint/state strategy.
- Some methods, especially AlphaEdit-style methods with large layer-stat/projection artifacts on GPU, may require larger GPUs than the base model size suggests.
- The checkpoint strategy should support full-state checkpoints first for reliability, while leaving room for method-specific partial-state checkpoints if implementation cost is acceptable.

## Implementation Priorities

- Move stable dataset builders and schema adapters out of notebooks into Python modules before major eval refactors.
- Remove `variant` from evaluation IO, visualization, and notebook configs.
- Update artifact loading to current HuggingFace/local config names: `es_loc` and `port`.
- Add alias-aware scoring before relying on `port` accepted answer lists.
- Then consider parallelizing judge repetitions or independent runs if judge throughput and scheduler capacity justify it.

## Current Implementation Status

- `EvalRecord` is alias-aware through list-only `ground_truths`.
- Result IO and visualization target the no-variant layout `data/evaluation/{model}/{milestones}/r{n_runs}-j{judge_repeats}/{method}/` with `perf/` and `time/` subfolders.
- `notebook/edit.ipynb` currently runs `es_loc` only from HuggingFace. `port` loading and execution are present but intentionally commented out until the rewritten portability artifact is ready.
- `notebook/eval.ipynb` runs the external judge worker from a separate notebook/kernel/process.
- `time/{result_name}.{run}.json` stores only `edit` per-step timing.
- `mem/{result_name}.{run}.json` stores edit-time peak CUDA memory when CUDA is available.
- `unscored/{result_name}.{run}.{step}.json` stores generated predictions at milestone state before judge scoring.
- `scored/{result_name}.{run}.{step}.json` stores judge-scored milestone snapshots.
- `perf/{result_name}.{run}.json` is rebuilt from available scored snapshots by the external judge worker.
- Process-level judge scoring from pending unscored records is implemented in `src/evaluate/judge_worker.py`.
- Judge worker scheduling now keeps up to `file_concurrency` active file-scoring tasks and rescans for new files every `poll_interval` while tasks are still running. New files can fill idle worker slots without waiting for the whole previously discovered batch to finish.
- When `methods` and `result_names` are explicit, the judge worker can compute the full expected scored-file grid: `methods * result_names * runs * milestones`. `judge_repeats` is not a file multiplier; it is averaged inside each scored file. With `stop_when_complete=True`, the worker exits once every expected `scored/{result_name}.{run}.{step}.json` exists.
- Judge requests dynamically inject vLLM-style structured output into `model_config.extra_body`: `structured_outputs.regex` is built per batch as exactly `n` binary lines. This should stay in code rather than static YAML because the final batch may be smaller than `judge.batch_size`.
- Edit logs use `data/evaluation/logs/edit-*.log`; judge worker and judge retry/status logs use `data/evaluation/logs/judge-*.log` by default. Keep edit and judge logs separate because they run in separate processes.
