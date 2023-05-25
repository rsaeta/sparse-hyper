import sys, os
from omegaconf import OmegaConf, DictConfig, MISSING
from config import RunConfig

import hydra
from hydra.core.config_store import ConfigStore

sys.path.append(os.path.abspath('..'))

from spalbp.models import ModelConfig, GeneratingTransformer
from utils import post_process_cfg


"""
Putting this here so I don't forget in the future. To override the type of attention mechanism
to use on a block of t-blocks, use the commandline override: 
model/t_blocks/attention@model._t_block_dict.t_block1.attention=entmax

To override the repeat parameter of a t-block, use the commandline override:
model._t_block_dict.t_block1.repeat=2

To override sub-parameters of a t-block, use the commandline override:
model._t_block_dict.t_block1.attention.alpha=1.2
"""


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
