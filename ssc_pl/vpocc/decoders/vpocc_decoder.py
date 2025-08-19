import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..layers import (ASPP, LearnableSqueezePositionalEncoding, TransformerLayer, Upsample,
                      Vanishing_point_based_cross_attention)

from ..utils import (cumprod, generate_grid, interpolate_flatten, pix2vox)

from .voxel_lifting import VoxelProposalLayerOrigin, VoxelProposalLayerWarp
from .voxel_fusion import VolumeFusion
from .unet3d import UNet3D
from termcolor import cprint

class VPOccDecoder(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_levels,
                 scene_shape,
                 project_scale,
                 image_shape,
                 voxel_size,
                 downsample_z,
                 num_layers=3,
                 num_heads=8,
                 num_points=9,
                 norm_layer=nn.BatchNorm3d,
                 bn_momentum=0.1):
        super().__init__()

        self.embed_dims = embed_dims
        scene_shape = [s // project_scale for s in scene_shape]
        
        if downsample_z != 1:
            self.ori_scene_shape = copy.copy(scene_shape)
            scene_shape[-1] //= downsample_z
        self.scene_shape = scene_shape
        self.num_queries = cumprod(scene_shape)
        self.image_shape = image_shape # (h, w)
        self.voxel_size = voxel_size * project_scale
        self.downsample_z = downsample_z

        self.voxel_proposal = VoxelProposalLayerOrigin(embed_dims, scene_shape, image_shape=image_shape, num_layers=num_layers, 
                                                       num_levels=num_levels, num_heads=num_heads, num_points=num_points)
        self.voxel_proposal_warp = VoxelProposalLayerWarp(embed_dims, scene_shape, num_layers=num_layers, 
                                                          num_levels=num_levels, num_heads=num_heads, num_points=num_points)

        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        
        self.scene_pos = LearnableSqueezePositionalEncoding((128, 128, 2),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))

        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)
        self.register_buffer('image_grid', image_grid)

        assert project_scale in (1, 2)
        
        self.occ_head= nn.Sequential(
            nn.Sequential(nn.ConvTranspose3d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=(1, 1, downsample_z),
                    padding=1,
                    output_padding=(0, 0, downsample_z - 1),
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU()) if downsample_z != 1 else nn.Identity(),
            ASPP(embed_dims, dilations_conv_list=(1, 2, 3)),
            Upsample(embed_dims, embed_dims, norm_layer, bn_momentum) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1))
        
        self.decoder = UNet3D(embed_dims, self.scene_shape, num_classes, norm_layer=norm_layer, bn_momentum=bn_momentum)
        self.voxel_fusion = VolumeFusion(embed_dims, embed_dims)
    
    def apply_homography(self, projected_pix, batch, fov_mask):
        """
        Apply homography to a set of 2D points using PyTorch.
        
        Parameters:
        ----------
        points: torch.Tensor
            2D points in shape (N, 2).
        H: torch.Tensor
            Homography matrix of shape (3, 3).
            
        Returns:
        -------
        warped_points: torch.Tensor
            Warped points of shape (N, 2).
        """
        # Ensure the inputs are on the same device (e.g., GPU)# torch.Size([1, 1, 1, 46592, 2])
        device = projected_pix.device
        fov_mask = fov_mask.squeeze(0)

        W = self.image_shape[1]
        H = self.image_shape[0]
        M_L = batch["M_left"].squeeze(0)
        M_R = batch["M_right"].squeeze(0)
        points = projected_pix.squeeze(0)

        # Convert to homogeneous coordinates
        ones = torch.ones(points.shape[0], 1, device=device)
        homogenous_points = torch.cat([points, ones], dim=1)
        
        # Apply homography
        warped_homogeneous_L = torch.mm(homogenous_points.clone(), M_L.t())
        warped_homogeneous_R = torch.mm(homogenous_points.clone(), M_R.t())
        
        # Convert back to cartesian coordinates
        warped_points_L = warped_homogeneous_L[:, :2] / warped_homogeneous_L[:, 2].unsqueeze(1)
        warped_points_R  = warped_homogeneous_R[:, :2] / warped_homogeneous_R[:, 2].unsqueeze(1)
        
        mask_L = (points[:, 0] > 0) & (points[:, 0] <= int(W/2)) & (points[:, 1] > 0) & (points[:, 1] < H)
        mask_R = (points[:, 0] > int(W/2)) & (points[:, 0] < W) & (points[:, 1] > 0) & (points[:, 1] < H)
        
        points[mask_L] = warped_points_L[mask_L]
        points[mask_R] = warped_points_R[mask_R]
        
        mask_H = (points[:, 1] > 0) & (points[:, 1] < H)
        mask_W = (points[:, 0] > 0) & (points[:, 0] < W)
        
        valid_mask = mask_H & mask_W
        fov_mask[~valid_mask] = False
    
        return points.unsqueeze(0).long(), fov_mask.unsqueeze(0)
    
    @autocast(dtype=torch.float32)
    def forward(self, vanishing_point, feats_dict, depth, K, E, voxel_origin, projected_pix,
                fov_mask, warp_dict):
        
        feats = feats_dict['origin_feats']
        warped_feats = feats_dict['warped_feats']
                
        bs = feats[0].shape[0]

        if self.downsample_z != 1:
            projected_pix = interpolate_flatten(
                projected_pix, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            fov_mask = interpolate_flatten(
                fov_mask, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            
            warped_projected_pix, warped_fov_mask = self.apply_homography(projected_pix.clone().float(), warp_dict, fov_mask.clone())

        vol_pts = pix2vox(
            self.image_grid,
            depth.unsqueeze(1),
            K,
            E,
            voxel_origin,
            self.voxel_size,
            downsample_z=self.downsample_z).long()

        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_pix = torch.flip(ref_pix, dims=[-1])
        
        warped_ref_pix = (torch.flip(warped_projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(warped_projected_pix)
        warped_ref_pix = torch.flip(warped_ref_pix, dims=[-1])
        
        origin_scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        warped_scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        
        scene_pos = self.scene_pos().repeat(bs, 1, 1)
        
        origin_scene_embed = self.voxel_proposal(vanishing_point, origin_scene_embed, feats, scene_pos, vol_pts, ref_pix)
        warped_scene_embed = self.voxel_proposal_warp(warped_scene_embed, warped_feats, scene_pos, vol_pts, warped_ref_pix)
        
        scene_embed = self.voxel_fusion(origin_scene_embed, warped_scene_embed)

        scene_embed = self.decoder(scene_embed)
        output = self.occ_head(scene_embed)

        return output