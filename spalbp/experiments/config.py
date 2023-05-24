from dataclasses import dataclass
from typing import List
from omegaconf import MISSING
import os
import sys

sys.path.append(os.path.abspath('..'))

from spalbp.models import ModelConfig


@dataclass
class OptimizerConfig:
    type: str
    lr: float
    betas: List[float]
    eps: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    type: str
    warmup_steps: int
    min_lr: float


@dataclass
class TrainingConfig:
    batch_size: int
    num_batches: int
    grad_clipping_value: int
    validation_every: int
    log_every: int
    save_every: int
    save_last: bool


@dataclass
class ExperimentConfig:
    type: str
    context_size: int
    data_source: str
    num_classes: int
    save_dir: str
    load_dir: str
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    save_last: bool
    wandb_project: str


@dataclass
class RunConfig:
    experiment: ExperimentConfig = MISSING
    model: ModelConfig = MISSING
