"""Train/infer pipeline entrypoints."""

from .infer import run_inference as run_inference
from .train import run_training as run_training

__all__ = ["run_training", "run_inference"]
