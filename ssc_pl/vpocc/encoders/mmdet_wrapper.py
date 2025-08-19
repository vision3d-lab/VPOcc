from importlib import import_module

import torch
import torch.nn as nn

from mmengine.config import Config
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS
from termcolor import cprint

class MMDetWrapper(nn.Module):

    def __init__(self,
                 config_path,
                 custom_imports,
                 checkpoint_path=None,
                 embed_dims=256,
                 scales=(4, 8, 16),
                 num_queries=100,
                 freeze=False):
        
        super().__init__()
        import_module(custom_imports)
        config = Config.fromfile(config_path)
        self.hidden_dims = config.model.panoptic_head.decoder.hidden_dim
        self.model = MODELS.build(config.model)
        self.scales = scales
        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu'))
            )  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM
        self.model.panoptic_head.predictor = None

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if embed_dims != self.hidden_dims:
            self.out_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.hidden_dims, embed_dims, 1),
                    nn.BatchNorm2d(embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dims, embed_dims, 1),
                ) for _ in scales
            ])

    def forward(self, x):
        # TODO: The following is only devised for the MaskDINO implementation.
        feats = self.model.extract_feat(x)
        
        mask_feat, _, multi_scale_feats = self.model.panoptic_head.pixel_decoder.forward_features(
            feats, masks=None)
        
        feats_list = (feats[0], *multi_scale_feats[:2])
        
        feats = []
        
        for i in self.scales:
            assert i in (4, 8, 16)
            if i == 4:
                feats.append(feats_list[0])
            elif i == 8:
                feats.append(feats_list[1])
            elif i == 16:
                feats.append(feats_list[2])
        
        # dimension matching
        if hasattr(self, 'out_projs'):
            feats = [proj(feat) for proj, feat in zip(self.out_projs, feats)]

        
        return dict(
            feats=feats)  # self.pts_embed = Embedding(100, 2)