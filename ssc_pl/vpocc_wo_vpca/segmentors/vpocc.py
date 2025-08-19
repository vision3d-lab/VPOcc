import torch.nn as nn
import torch
from ... import build_from_configs
from .. import encoders
from ..decoders import VPOccDecoder
from ..losses import ce_ssc_loss, geo_scal_loss, sem_scal_loss
from .side_warping_symphonies import SideWarping

class VPOcc(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        image_shape=None,
        voxel_size=None,
        downsample_z=None,
        class_weights=None,
        criterions=None,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        self.decoder = VPOccDecoder(
            embed_dims,
            num_classes,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            downsample_z=downsample_z,
    )
        self.side_warping = SideWarping(image_shape=image_shape)
        
    def forward(self, inputs):
        warped_img, warp_dict = self.side_warping(inputs['img'], inputs)
        
        warped_pred_insts = self.encoder(warped_img)
        warped_feats = warped_pred_insts.pop('feats')
        
        origin_pred_insts = self.encoder(inputs['img'])
        origin_feats = origin_pred_insts.pop('feats')
        
        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))

        feats_dict = {'origin_feats': origin_feats, 'warped_feats': warped_feats}
        vanishing_point = inputs['v_pts'][0,:]
        
        output = self.decoder(vanishing_point, feats_dict, depth, K, E, voxel_origin, projected_pix, fov_mask, warp_dict)
        return {'ssc_logits': output}

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        losses = {}
        
        for loss in self.criterions:
            losses['loss_' + loss] = loss_map[loss](preds, target)

        return losses