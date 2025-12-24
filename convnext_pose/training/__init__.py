"""Training components for ConvNeXt-Pose."""

from .config import build_config, AVAILABLE_BACKBONES
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    "build_config",
    "AVAILABLE_BACKBONES",
    "Trainer",
    "Evaluator",
]
