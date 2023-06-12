import os
import torch
import wandb

import transformers
from config import RunConfig
from datasets import load_dataset
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from utils import (
    post_process_cfg,
    init_wandb,
    get_tokenizer,
    get_model,
    get_criterion,
    save_model,
    setup,
    learners,
    device,
    MaskedLMDataset,

)
from synthetic_mask import random_sample_data2

cs = ConfigStore.instance()
cs.store(name='run', node=RunConfig)

def _train(cfg: RunConfig):
    model = get_model(cfg)
    teacher_model = get_teacher_model(cfg)


