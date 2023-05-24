from dataclasses import dataclass, field
import sys, os
from omegaconf import OmegaConf, DictConfig, MISSING

import hydra
from hydra.core.config_store import ConfigStore

sys.path.append(os.path.abspath('..'))

from spalbp.models import ModelConfig, TransformerModel


@dataclass
class ExperimentConfig:
    context_size: int = 256


@dataclass
class RunConfig:
    experiment: ExperimentConfig = MISSING
    model: ModelConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="run", node=RunConfig)


@hydra.main(version_base=None, config_path="../config", config_name="sparse_model")
def main(cfg: RunConfig):

    cfg = OmegaConf.structured(RunConfig(**cfg))
    model = TransformerModel(cfg.model)
    breakpoint()
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == '__main__':
    main()

