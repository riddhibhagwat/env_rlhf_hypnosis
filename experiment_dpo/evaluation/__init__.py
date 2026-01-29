"""Evaluation modules for DPO experiments."""

from .eval_mcq import evaluate_mcq
from .eval_generation import evaluate_generation

__all__ = [
    "evaluate_mcq",
    "evaluate_generation"
]
