import sys, os
from omegaconf import OmegaConf, DictConfig, MISSING
from config import RunConfig

import hydra
from hydra.core.config_store import ConfigStore

sys.path.append(os.path.abspath('..'))

from spalbp.models import ModelConfig, GeneratingTransformer
from utils import post_process_cfg


cs = ConfigStore.instance()
cs.store(name="run", node=RunConfig)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: OmegaConf):
    cfg = post_process_cfg(cfg)
    breakpoint()

    model = GeneratingTransformer(cfg.model)

    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == '__main__':
    main()
