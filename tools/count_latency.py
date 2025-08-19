import hydra
from fvcore.nn import FlopCountAnalysis, flop_count_table
from omegaconf import DictConfig, open_dict

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks
import os
import os.path as osp

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks
from tqdm import tqdm


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    cfg, _ = pre_build_callbacks(cfg)
    
    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[-1]
    output_dir = osp.join('outputs', cfg.data.datasets.type)
    
    if cfg.trainer.devices != 1:
        with open_dict(cfg.trainer):
            cfg.trainer.devices = 1
    
    model = LitModule(**cfg, meta_info=meta_info)
    model.cuda()
    model.eval()
    
    latency_list = []
    
    # GPU warm up
    i = 0
    with torch.no_grad():
        for batch_inputs, targets in track(data_loader):
            if i == 0:
                #GPU-WARM-UP
                print("@@@@ GPU-WARM-UP @@@@")
                with torch.no_grad():
                    for i in tqdm(range(200)):
                        _ = model(batch_inputs)
                print("@@@@ GPU-WARM-UP-DONE @@@@")
                
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()  # 시작 이벤트 기록

            outputs = model(batch_inputs)
            
            end.record()  # 종료 이벤트 기록
            torch.cuda.synchronize()  # GPU 작업 완료를 기다림

            latency = start.elapsed_time(end)
            latency_list.append(latency)
            print("@@ latency: ", latency, "ms")
            i += 1
    num_samples = len(latency_list)
    latency = sum(latency_list) / num_samples / 1000
    print("@@@ Mean latency: ", latency, "s")
    
    throughput = num_samples / latency * 1000
    print("@@@ Mean throughput: ", throughput, "img/s")

    log_str = f'latency list: {latency_list}\n' \
        f"latency(s): {latency}, throughput(img/s): {throughput}\n"

    file_path = os.path.join(self.save_dir, f'latency.log')
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write(log_str)
    else:
        with open(file_path, 'a') as file:
            file.write(log_str)



if __name__ == '__main__':
    main()
