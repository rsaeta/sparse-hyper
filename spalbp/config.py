from dataclasses import dataclass
from typing import List
from omegaconf import MISSING

from lib.models import ModelConfig


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
class TokenizerConfig:
    type: str
    train_file: str
    load_file: str
    vocab_size: int


@dataclass
class DataConfig:
    name: str
    batch_size: int
    num_classes: int
    output_type: str
    tokenizer: TokenizerConfig


@dataclass
class ExperimentConfig:
    type: str
    context_size: int
    num_classes: int
    save_dir: str
    load_dir: str
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    data: DataConfig
    save_last: bool
    wandb_project: str


@dataclass
class RunConfig:
    experiment: ExperimentConfig = MISSING
    model: ModelConfig = MISSING
