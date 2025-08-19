import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks
from lightning.pytorch import seed_everything
import torch
import numpy as np
import random

@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, workers=True)
    
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, **meta_info)
    checkpoint_path = cfg.checkpoint_path
    trainer = L.Trainer(**cfg.trainer, **callbacks, resume_from_checkpoint=checkpoint_path)
    trainer.fit(model, *dls[:2])

if __name__ == '__main__':
    main()