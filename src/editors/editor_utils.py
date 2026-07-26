from dataclasses import dataclass


@dataclass
class EditRequest:
    subject: str
    prompt: str
    answer: str
