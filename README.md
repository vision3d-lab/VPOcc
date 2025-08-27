
# [IROS 2025] VPOcc: Exploiting Vanishing Point for 3D Semantic Occupancy Prediction

[![Paper](https://img.shields.io/badge/arXiv-2408.03551-b31b1b.svg)](https://arxiv.org/abs/2408.03551)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://vision3d-lab.github.io/vpocc/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/papers/2408.03551)

This repository contains the official implementation of **VPOcc: Exploiting Vanishing Point for 3D Semantic Occupancy Prediction**.

## 📋 Contents
1. [Installation](#1-installation)
2. [Preparing Dataset](#2-preparing-dataset)
3. [Results](#3-results)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Test](#6-test)
7. [Visualization](#7-visualization)
8. [Acknowledgements](#8-acknowledgements)
9. [BibTeX](#9-bibtex)

## 1. Installation
Our code is based on **CUDA 11.3** and **PyTorch 1.11.0**.

a. Download the source code:
```shell
git clone https://github.com/vision3d-lab/VPOcc.git
cd VPOcc
```

b. Create conda environment and install PyTorch 1.11.0 with CUDA 11.3:
```shell
conda create -n vpocc python=3.8 -y
conda activate vpocc
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

c. Install MMCV and MMDet with OpenMIM:
```shell  
pip install -U openmim
mim install mmengine==0.9.0 
mim install mmcv==2.0.1
mim install mmdet==3.2.0
```

d. Install additional requirements:
```shell
pip install -r requirements.txt
```

e. Initialize environment:
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
wandb init
```

## 2. Preparing Dataset

We follow the steps from [Symphonies](https://github.com/hustvl/Symphonies?tab=readme-ov-file).

a. **Dataset**
- **SemanticKITTI**: Download RGB images, calibration files, and preprocess the labels (refer to [VoxFormer](https://github.com/NVlabs/VoxFormer/blob/main/docs/prepare_dataset.md) or [MonoScene](https://github.com/astra-vision/MonoScene#semantickitti) documentation).  
- **SSCBench-KITTI360**: Refer to [SSCBench-KITTI360 Dataset Guide](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360).

b. **Depth Prediction**
- **SemanticKITTI**: Generate depth predictions with pre-trained MobileStereoNet (see [VoxFormer Preprocess Guide](https://github.com/NVlabs/VoxFormer/tree/main/preprocess#3-image-to-depth)).  
- **SSCBench-KITTI360**: Follow the same procedure as SemanticKITTI, but adapt disparity values following [issue](https://github.com/ai4ce/SSCBench/issues/8#issuecomment-1674607576).

c. **Vanishing Point Extraction**
- Use pre-trained [NeurVPS](https://github.com/zhou13/neurvps) to extract vanishing points.  
- Download pre-extracted vanishing points and extraction code from [Hugging Face Dataset](https://huggingface.co/datasets/joonsu0109/vpocc-vanishing-points/tree/main).

d. **Data Structure (Softlink under `./data`)**
```
./data
├── SemanticKITTI
│   ├── dataset 
│   ├── labels
│   ├── depth
│   └── vanishing_points
└── SSCBench-KITTI360
    ├── data_2d_raw
    ├── depth
    ├── monoscene_preprocess
    └── vanishing_points
```

e. **Pretrained Weights**
- [Pre-trained MaskDINO](https://huggingface.co/joonsu0109/vpocc-symphonies-maskdino) → place under `./backups`  
```
./VPOcc
├── backups
├── ckpts
├── configs
├── data
├── maskdino
├── outputs
├── ssc_pl
└── tools
```

## 3. Results

| Dataset             | Validation (IoU / mIoU)                                                                                                  | Test (IoU / mIoU)                                                                                             | Model                                                                                  |
|---------------------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **SemanticKITTI**   | 44.98 / 16.36 [📄 log](https://huggingface.co/joonsu0109/vpocc-semantickitti/blob/main/training_metrics.log)              | 44.58 / 16.15 [📄 log](https://huggingface.co/joonsu0109/vpocc-semantickitti/blob/main/test_log.txt)           | [model](https://huggingface.co/joonsu0109/vpocc-semantickitti)                      |
| **SSCBench-KITTI360** | 46.35 / 20.06 [📄 log](https://huggingface.co/joonsu0109/vpocc-sscbench-kitti360/blob/main/training_metrics.log)        | 46.39 / 19.80 [📄 log](https://huggingface.co/joonsu0109/vpocc-sscbench-kitti360/blob/main/test_metrics.log)  | [model](https://huggingface.co/joonsu0109/vpocc-sscbench-kitti360)                  |

## 4. Training

a. **SemanticKITTI**
```shell
python tools/train.py --config-name config.yaml trainer.devices=4 \
+data_root=./data/SemanticKITTI \
+label_root=./data/SemanticKITTI/labels \
+depth_root=./data/SemanticKITTI/depth \
+log_name=train_semantickitti \
+model_name=vpocc \
+seed=53
```

b. **KITTI-360**
```shell
python tools/train.py --config-name config_kitti_360.yaml trainer.devices=4 \
+data_root=./data/SSCBench-KITTI360 \
+label_root=./data/SSCBench-KITTI360/monoscene_preprocess/labels \
+depth_root=./data/SSCBench-KITTI360/depth \
+log_name=train_kitti360 \
+model_name=vpocc \
+seed=53
```

## 5. Evaluation

a. **SemanticKITTI**
```shell
python tools/evaluate.py --config-name config.yaml trainer.devices=1 \
+ckpt_path=./ckpts/semantickitti.ckpt \
+data_root=./data/SemanticKITTI \
+label_root=./data/SemanticKITTI/labels \
+depth_root=./data/SemanticKITTI/depth \
+log_name=eval_semantickitti \
+model_name=vpocc \
+seed=53
```

b. **KITTI-360**
```shell
python tools/evaluate.py --config-name config_kitti_360.yaml trainer.devices=1 \
+ckpt_path=./ckpts/kitti360.ckpt \
+data_root=./data/SSCBench-KITTI360 \
+label_root=./data/SSCBench-KITTI360/monoscene_preprocess/labels \
+depth_root=./data/SSCBench-KITTI360/depth \
+log_name=eval_kitti360 \
+model_name=vpocc \
+seed=53
```

## 6. Test

a. **SemanticKITTI** (hidden test set)
```shell
python tools/test_semantickitti.py --config-name config.yaml trainer.devices=1 \
+ckpt_path=./ckpts/semantickitti.ckpt \
+data_root=./data/SemanticKITTI \
+label_root=./data/SemanticKITTI/labels \
+depth_root=./data/SemanticKITTI/depth \
+log_name=test_semantickitti \
+model_name=vpocc \
+seed=53
```

b. **KITTI-360**
```shell
python tools/test_kitti360.py --config-name config_kitti_360.yaml trainer.devices=1 \
+ckpt_path=./ckpts/kitti360.ckpt \ \
+data_root=./data/SSCBench-KITTI360 \
+label_root=./data/SSCBench-KITTI360/monoscene_preprocess/labels \
+depth_root=./data/SSCBench-KITTI360/depth \
+log_name=test_kitti360 \
+model_name=vpocc \
+seed=53
```

## 7. Visualization
Outputs of the validation set are saved in `./outputs`.

a. **SemanticKITTI**
```shell
python tools/generate_outputs.py --config-name config.yaml trainer.devices=1 \
+ckpt_path=./ckpts/semantickitti.ckpt \
+data_root=./data/SemanticKITTI \
+label_root=./data/SemanticKITTI/labels \
+depth_root=./data/SemanticKITTI/depth \
+log_name=vis_semantickitti \
+model_name=vpocc
```

b. **KITTI-360**
```shell
python tools/generate_outputs.py --config-name config_kitti360.yaml trainer.devices=1 \
+ckpt_path=./ckpts/kitti360.ckpt \
+data_root=./data/SSCBench-KITTI360 \
+label_root=./data/SSCBench-KITTI360/monoscene_preprocess/labels \
+depth_root=./data/SSCBench-KITTI360/depth \
+log_name=vis_kitti360 \
+model_name=vpocc
```

## 8. Acknowledgements
Special thanks to [Symphonies](https://github.com/hustvl/Symphonies) and many thanks to the following excellent projects:
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [VoxFormer](https://github.com/NVlabs/VoxFormer)
- [MaskDINO](https://github.com/IDEA-Research/MaskDINO)
- [3D-CVF](https://github.com/rasd3/3D-CVF)
- [NeurVPS](https://github.com/zhou13/neurvps)

## 9. BibTeX
- Comming soon :D