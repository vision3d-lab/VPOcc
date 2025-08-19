"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ..layers import MPAC, ASPP


class VolumeFusion(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(VolumeFusion, self).__init__()

        self.origin_reduct_net = nn.Sequential(
            nn.Conv3d(in_channels=input_feature*2, out_channels=input_feature, kernel_size=3, stride=1,  padding=1,bias=False),
            nn.BatchNorm3d(input_feature),
            nn.ReLU(),
            MPAC(channel=input_feature, residual=True),
        )

        self.warping_reduct_net = nn.Sequential(
            nn.Conv3d(in_channels=input_feature*2, out_channels=input_feature, kernel_size=3, stride=1,  padding=1,bias=False),
            nn.BatchNorm3d(input_feature),
            nn.ReLU(),
            MPAC(channel=input_feature, residual=True),
        )

        self.fusion_refine = nn.Sequential(
            nn.Conv3d(in_channels=input_feature*2, out_channels=output_feature, kernel_size=3, stride=1,  padding=1,bias=False),
            nn.BatchNorm3d(output_feature),
            nn.ReLU(),
            MPAC(channel=output_feature, residual=True),
            nn.BatchNorm3d(output_feature),
            nn.ReLU(),
        )

    def forward(self, origin, warping):
        concat_map = torch.cat([origin, warping],  dim=1)
        fused_origin = torch.relu(self.origin_reduct_net(concat_map)) * origin
        fused_warping = torch.relu(self.warping_reduct_net(concat_map)) * warping
        return self.fusion_refine(torch.cat([fused_origin, fused_warping], dim=1))