import torch
import torch.nn as nn

from ..layers import (DeformableTransformerLayer, TransformerLayer,
                      Vanishing_point_based_cross_attention, CustomMultiScaleDeformableAttention)

from ..utils import (flatten_multi_scale_feats,
                     get_level_start_index, index_fov_back_to_voxels, nlc_to_nchw)

class VoxelProposalLayerOrigin(nn.Module):
    def __init__(self, embed_dims, scene_shape, image_shape=None, num_layers=None, num_levels=None, num_heads=None, num_points=None):
        super().__init__()
        
        self.attn = DeformableTransformerLayer(embed_dims, 
        num_heads=num_heads, 
        num_levels=num_levels,
        num_points=num_points,
        attn_layer=Vanishing_point_based_cross_attention,
        grid_size=image_shape
        )
        self.num_layers = num_layers
        self.scene_shape = scene_shape
        self.image_shape = image_shape
        
    def forward(self,vanishing_point, scene_embed, feats, scene_pos=None, vol_pts=None, ref_pix=None):

        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))
        assert vol_pts.shape[0] == 1
        geom = vol_pts.squeeze()[keep.squeeze()]

        pts_mask = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool)
        pts_mask[geom[:, 0], geom[:, 1], geom[:, 2]] = True
        pts_mask = pts_mask.flatten()

        feat_flatten, shapes = flatten_multi_scale_feats(feats)
        pts_embed = scene_embed[:, pts_mask]
        
        for i in range(self.num_layers):
            pts_embed = self.attn(
                pts_embed, 
                feat_flatten,  
                vanishing_point=vanishing_point,
                query_pos=scene_pos[:, pts_mask] if scene_pos is not None else None,    
                ref_pts=ref_pix[:, pts_mask].unsqueeze(2).expand(-1, -1, len(feats), -1),
                spatial_shapes=shapes,
                level_start_index=get_level_start_index(shapes)
                )
            
        return index_fov_back_to_voxels(
            nlc_to_nchw(scene_embed, self.scene_shape), pts_embed, pts_mask)

class VoxelProposalLayerWarp(nn.Module):
    def __init__(self, embed_dims, scene_shape, num_layers=None, num_levels=None, num_heads=None, num_points=None):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims,
                                               num_heads=num_heads, 
                                               num_levels=num_levels,
                                               num_points=num_points,
                                               attn_layer=CustomMultiScaleDeformableAttention)
        self.scene_shape = scene_shape
        self.num_layers = num_layers

    def forward(self, scene_embed, feats, scene_pos, vol_pts, ref_pix):
        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))
        assert vol_pts.shape[0] == 1
        geom = vol_pts.squeeze()[keep.squeeze()]

        pts_mask = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool)
        pts_mask[geom[:, 0], geom[:, 1], geom[:, 2]] = True
        pts_mask = pts_mask.flatten()

        feat_flatten, shapes = flatten_multi_scale_feats(feats)
        
        pts_embed = scene_embed[:, pts_mask]
        
        for i in range(self.num_layers):
            pts_embed = self.attn(
                pts_embed,
                feat_flatten,
                query_pos=scene_pos[:, pts_mask] if scene_pos is not None else None,
                ref_pts=ref_pix[:, pts_mask].unsqueeze(2).expand(-1, -1, len(feats), -1),
                spatial_shapes=shapes,
                level_start_index=get_level_start_index(shapes))

        return index_fov_back_to_voxels(
            nlc_to_nchw(scene_embed, self.scene_shape), pts_embed, pts_mask)