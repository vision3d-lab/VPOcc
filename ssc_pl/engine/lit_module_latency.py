import lightning as L
import torch.nn as nn
import torch.optim as optim
from omegaconf import open_dict
import torch 
from .. import build_from_configs, evaluation # , models
import importlib
import sys
import os 
import time
import psutil

class LitModule(L.LightningModule):

    def __init__(self, *, model, optimizer, scheduler, model_name, log_name, criterion=None, evaluator=None,  **kwargs):
        super().__init__()

        models = self.import_models(model_name)
        
        self.model = build_from_configs(models, model, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = build_from_configs(nn, criterion) if criterion else self.model.loss
        self.train_evaluator = build_from_configs(evaluation, evaluator)
        self.test_evaluator = build_from_configs(evaluation, evaluator)
        self.save_dir = os.path.join("outputs", log_name)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']
        self.latency = []
        
    def import_models(self, model_folder):
        # Assuming 'ssc_pl' is your package name and 'engine' is the subpackage
        package_name = 'ssc_pl.engine'  
        module = importlib.import_module(f"..{model_folder}", package=package_name)
        return module
        
    def forward(self, x):
        return self.model(x)

    def _step(self, batch, evaluator=None, mode=None):
        x, y = batch
        # test time 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        pred = self(x)
        
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)
        
        self.latency.append(latency)
        
        print("@@ latency: ", latency, "ms")
                    
        loss = self.criterion(pred, y)
        
        if evaluator:
            evaluator.update(pred, y)
            
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_evaluator)
        self.log('train/loss', {'loss_total': sum(loss.values()), **loss})
        self.log('train/loss_total', sum(loss.values()), sync_dist=True)

        return sum(list(loss.values())) if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx):      
        if batch_idx == 0:
            #GPU-WARM-UP
            print("@@@@@ GPU-WARM-UP @@@@@")
            self.model.eval()
            with torch.no_grad():
                for i in range(200):
                    x, y = batch
                    _ = self(x)
                    
            torch.cuda.synchronize() 

            print("@@@@@ GPU-WARM-UP-DONE @@@@@")
            
        self._shared_eval(batch, 'test')

    def _shared_eval(self, batch, prefix):
        loss = self._step(batch, self.test_evaluator, mode=prefix)
        # Lightning automatically accumulates the metric and averages it
        # if `self.log` is inside the `validation_step` and `test_step`
        self.log(f'{prefix}/loss', loss, sync_dist=True)
        self.log(f'{prefix}/loss_total', sum(loss.values()), sync_dist=True)

    def on_train_epoch_end(self):
        self._log_metrics(self.train_evaluator, 'train')

    def on_validation_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'val')

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_evaluator, 'test')

    def _log_metrics(self, evaluator, prefix=None):
        metrics = evaluator.compute()
        iou_per_class = metrics.pop('iou_per_class')
        if prefix:
            metrics = {'/'.join((prefix, k)): v for k, v in metrics.items()}

        self.log_dict(metrics, sync_dist=True)

        if hasattr(self, 'class_names'):
            self.log(
                prefix + '_iou_per_cls',
                {c: s.item()
                 for c, s in zip(self.class_names, iou_per_class)},
                sync_dist=True)
            
        if self.trainer.is_global_zero and prefix == 'val':
                print(metrics)
                print("class_names: ", self.class_names)
                print("iou_per_class: ", [f"{i.item()*100:.2f}" for i in iou_per_class])
                
                log_str = f"epoch: {self.current_epoch}\n" \
                  f"{metrics}\n" \
                  f"class_names: {self.class_names}\n" \
                  f"iou_per_class: {[f'{i.item()*100:.2f}' for i in iou_per_class]}\n"
                  
                file_path = os.path.join(self.save_dir, f'metrics.log')
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as file:
                        file.write(log_str)
                else:
                    with open(file_path, 'a') as file:
                        file.write(log_str)
                
        if prefix == 'test':
                print(metrics)
                print("class_names: ", self.class_names)
                print("iou_per_class: ", [f"{i.item()*100:.2f}" for i in iou_per_class])

                num_samples = len(self.latency)
                latency = sum(self.latency) / num_samples / 1000
                print("@@@ Mean latency: ", latency, "s")
                
                throughput = num_samples / sum(self.latency) * 1000
                print("@@@ Mean throughput: ", throughput, "img/s")
                
                # calculate std and var
                std = torch.std(torch.tensor(self.latency))
                var = torch.var(torch.tensor(self.latency))
                print("@@@ std: ", std)
                print("@@@ var: ", var)
            
                log_str = f"epoch: {self.current_epoch}\n" \
                        f"{metrics}\n" \
                        f"class_names: {self.class_names}\n" \
                        f"iou_per_class: {[f'{i.item()*100:.2f}' for i in iou_per_class]}\n" \
                        f"latency(s): {latency}, throughput(img/s): {throughput}\n" \
                        f"std: {std}, var: {var}\n" \
                        f"num_samples: {num_samples}\n" \
                        f"latency_list: {self.latency}\n"
                
                  
                
                file_path = os.path.join(self.save_dir, f'metrics.log')
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as file:
                        file.write(log_str)
                else:
                    with open(file_path, 'a') as file:
                        file.write(log_str)
                        
        evaluator.reset()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer
        scheduler_cfg = self.scheduler
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg]

            for k, v in self.named_parameters():
                for i, pg_cfg in enumerate(paramwise_cfg):
                    if 'name' in pg_cfg and pg_cfg.name in k:
                        pgs[i].append(v)
                    # USER: Customize more cfgs if needed
                    else:
                        params.append(v)
        else:
            params = self.parameters()
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        if paramwise_cfg:
            for pg, pg_cfg in zip(pgs, paramwise_cfg):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg.lr * pg_cfg.lr_mult
                # USER: Customize more cfgs if needed
                optimizer.add_param_group({'params': pg, **cfg})
        scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg.interval}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}