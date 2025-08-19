from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning.pytorch.callbacks import Callback

from .tabular_logger import TabularLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichProgressBar

import os 
import wandb
import torch 

class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        
        items.pop("v_num", None)
        items.pop("loss", None)
        
        loss_total = trainer.callback_metrics.get('train/loss_total')
        if loss_total is not None:
            loss_total = round(loss_total.item(), 3) 
        items['train/loss_total'] = loss_total
        
        val_loss = trainer.callback_metrics.get('val/loss_total')
        if val_loss is not None:
            val_loss = round(val_loss.item(), 3)  
        items['val/loss_total'] = val_loss
        
        val_iou = trainer.callback_metrics.get('val/IoU')
        if val_iou is not None:
            val_iou = round(val_iou.item(), 3)  
        items['val/IoU'] = val_iou
        
        val_miou = trainer.callback_metrics.get('val/mIoU')
        if val_miou is not None:
            val_miou = round(val_miou.item(), 3)  
        items['val/mIoU'] = val_miou
        
        return items


def pre_build_callbacks(cfg: DictConfig):
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = None

    if cfg.get('data_root'):
        cfg.data.datasets.data_root = cfg.data_root
    if cfg.get('label_root'):
        cfg.data.datasets.label_root = cfg.label_root
    if cfg.get('depth_root'):
        cfg.data.datasets.depth_root = cfg.depth_root
        
    save_dir = os.path.join("outputs", cfg.log_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = [WandbLogger(project="vpocc",name=f"{cfg.log_name}")]
    callback = [
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename='e{epoch}_miou_{val/mIoU:.4f}',
            monitor='val/mIoU',
            mode='max',
            auto_insert_metric_name=False,
            save_top_k=3
            ),
        callbacks.ModelSummary(max_depth=3),
        CustomRichProgressBar(),
        ]

    return cfg, dict(logger=logger, callbacks=callback)


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
