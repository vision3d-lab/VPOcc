import torch.nn as nn
from ..layers import Downsample, Upsample, MPAC, ASPP
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .voxel_fusion import VolumeFusion

class UNet3D(nn.Module):

    def __init__(
        self,
        channels,
        scene_size,
        num_classes,
        num_relations=4,
        project_scale=1,
        context_prior=True,
        norm_layer=None,
        bn_momentum=None
    ):
        super().__init__()
        feature_l1 = channels
        feature_l2 = feature_l1 * 2
        feature_l3 = feature_l2 * 2
        scene_size_l3 = [int(s / 4 / project_scale) for s in scene_size]

        self.process_l1 = nn.Sequential(
            Downsample(feature_l1, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Downsample(feature_l2, norm_layer, bn_momentum),
        )
        
        self.up_13_l2 = Upsample(feature_l3, feature_l2, norm_layer, bn_momentum)
        self.up_12_l1 = Upsample(feature_l2, feature_l1, norm_layer, bn_momentum)
  
        self.bottleneck = ASPP(feature_l3, (1, 2, 3))

        if project_scale != 1:
            self.up_l1_full = Upsample(channels, channels // 2, norm_layer, bn_momentum)
            channels = channels // 2
        else:
            self.up_l1_full = nn.Identity()

    def forward(self, x):
        x3d_l1 = x
        x3d_l2 = self.process_l1(x3d_l1)
        x3d_l3 = self.process_l2(x3d_l2)
    

        x3d_l3 = self.bottleneck(x3d_l3)
        
        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_full = self.up_l1_full(x3d_up_l1)

        return x3d_full