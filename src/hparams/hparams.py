from dataclasses import dataclass


"""
Several params take dot-notation paths into the model's layer hierarchy

To list all valid paths for a loaded model:
    for path, module in model.named_modules():
        print(path, '->', type(module).__name__)
"""
@dataclass
class HyperParams:
    # Name of the editing algorithm
    alg_name: str

    # HuggingFace model ID or local path to the model directory
    model_name: str

    # CUDA device index to load the model on
    gpu_id: int

    # When True, load model in bfloat16 instead of float32
    fp16: bool
