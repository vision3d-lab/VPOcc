import numpy as np
import cv2
import os
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
from torchvision import transforms
import kornia as K

class SideWarping(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.alpha = 0.2
        self.image_shape = image_shape
    
    def img_warp(self, imgs, v_pts):
        h, w = self.image_shape
        B = v_pts.shape[0]
        device = imgs.device
        ratio = self.alpha

        small_box_width = (w * ratio) / 2
        small_box_height = (h * ratio) / 2

        center_w = int(w / 2)
        center_h = int(h / 2)
        
        src_pts_right = torch.tensor([[
            [v_pts[:,0], v_pts[:,1] - small_box_height],
            [v_pts[:,0], v_pts[:,1] + small_box_height],
            [w, h],
            [w, 0]
        ]], device=device, dtype=torch.float32).repeat(B, 1, 1)

        src_pts_left = torch.tensor([[
            [v_pts[:,0], v_pts[:,1] - small_box_height],
            [v_pts[:,0], v_pts[:,1] + small_box_height],
            [0, h],
            [0, 0]
        ]], device=device, dtype=torch.float32).repeat(B, 1, 1)

        center_point_upper = (v_pts[:,1] - small_box_height + 0) / 2
        center_point_lower = (v_pts[:,1] + small_box_height + h) / 2

        dst_pts_right = torch.tensor([[
            [center_w, center_point_upper],
            [center_w, center_point_lower],
            [w, center_point_lower],
            [w, center_point_upper]
        ]], device=device, dtype=torch.float32).repeat(B, 1, 1)

        dst_pts_left = torch.tensor([[
            [center_w, center_point_upper],
            [center_w, center_point_lower],
            [0, center_point_lower],
            [0, center_point_upper]
        ]], device=device, dtype=torch.float32).repeat(B, 1, 1)

        
        M_left = K.geometry.get_perspective_transform(src_pts_left, dst_pts_left)
        
        M_right = K.geometry.get_perspective_transform(src_pts_right, dst_pts_right)
        
        img_warp_left: torch.tensor = K.geometry.warp_perspective(imgs, M_left, dsize=(h, w))
        img_warp_right: torch.tensor = K.geometry.warp_perspective(imgs, M_right, dsize=(h, w))
        
        img_warp = imgs.clone()
        img_warp[:,:,:,:int(center_w)] = img_warp_left[:,:,:,:int(center_w)]
        img_warp[:,:,:,int(center_w):] = img_warp_right[:,:,:,int(center_w):]
        
        return M_left, M_right, img_warp, src_pts_left, src_pts_right, dst_pts_left, dst_pts_right


    def forward(self, imgs, img_metas):
        v_pts = img_metas['v_pts']
        M_left, M_right, img_warp, pt_src_left, pt_src_right, pt_dst_left, pt_dst_right = self.img_warp(imgs, v_pts) # left
        warp_dict = {
            "M_left": M_left,
            "M_right": M_right,
            "pt_src_left": pt_src_left,
            "pt_src_right": pt_src_right,
            "pt_dst_left": pt_dst_left,
            "pt_dst_right": pt_dst_right,
        }
        return img_warp, warp_dict