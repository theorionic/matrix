"""DWA — Dynamic Weight Assembly model."""

from .assembly import WeightAssembler
from .config import DWAConfig, TrainConfig
from .losses import aux_losses, task_loss
from .model import DWAModel, forward_and_loss
from .parts import PartA, PartB
from .pool import VectorPool
from .retrieval import MultiAspectRetrieval
from .schedule import PhaseScheduler

__all__ = [
    "DWAConfig",
    "TrainConfig",
    "DWAModel",
    "VectorPool",
    "MultiAspectRetrieval",
    "WeightAssembler",
    "PartA",
    "PartB",
    "PhaseScheduler",
    "aux_losses",
    "task_loss",
    "forward_and_loss",
]
