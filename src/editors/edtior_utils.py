from dataclasses import dataclass


@dataclass
class EditRequest:
    prompt: str
    old_ground_truth: str
    new_ground_truth: str
