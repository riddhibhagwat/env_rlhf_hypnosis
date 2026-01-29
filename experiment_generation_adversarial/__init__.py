"""
Experiment: Generation-Based Adversarial Testing

This package provides tools to test whether models trained with poisoned RLHF data
generate adversarial/false outputs in free-form responses, as opposed to just 
selecting adversarial options in multiple-choice questions.

Main modules:
- experiment_generation_adversarial_outputs: Core experiment logic
- pipeline_generation_experiment: End-to-end pipeline

Usage:
    See README.md for detailed usage instructions.
"""

__version__ = "1.0.0"
__author__ = "RLHF Research Team"

# Make key functions available at package level
try:
    from .experiment_generation_adversarial_outputs import (
        generate_response,
        evaluate_response,
        run_evaluation,
    )
except ImportError:
    # Allow imports to fail gracefully if dependencies aren't installed
    pass

