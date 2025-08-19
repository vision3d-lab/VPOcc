import glob
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from ...utils.helper import vox2pix
import json
from termcolor import cprint

SPLITS = {
    'train': ('00', '01', '02', '03', '04', '05', '06', '07', '09', '10'),
    'val': ('08', ),
    'test': ('11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'),
}

SEMANTIC_KITTI_CLASS_FREQ = torch.tensor([
    5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05, 8.21951000e05,
    2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07, 4.50296100e06, 4.48836500e07,
    2.26992300e06, 5.68402180e07, 1.57196520e07, 1.58442623e08, 2.06162300e06, 3.69705220e07,
    1.15198800e06, 3.34146000e05
])


class SemanticKITTI(Dataset):

    META_INFO = {
        'class_weights':
        1 / torch.log(SEMANTIC_KITTI_CLASS_FREQ + 1e-6),
        'class_names':
        ('empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
         'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
         'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')
    }

    def __init__(
        self,
        split,
        data_root,
        label_root,
        num_classes=None,
        depth_root=None,
        project_scale=None,
        flip=True,
        load_pose=False,
        image_shape=None,
        scene_size=None,
        voxel_size=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.sequences = SPLITS[split]
        self.split = split

        self.depth_root = depth_root
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.flip = flip
        self.load_pose = load_pose
        self.num_classes = num_classes

        self.voxel_origin = np.array((0, -25.6, -2))
        self.voxel_size = voxel_size
        self.scene_size = scene_size
        self.img_shape = image_shape
        self.scans = []

        for sequence in self.sequences:
            sequence_path = osp.join(self.data_root, 'dataset', 'sequences', sequence)
            calib = self.read_calib(osp.join(sequence_path, 'calib.txt'))
            P = calib['P2']
            T_velo_2_cam = calib['Tr']
            proj_matrix = P @ T_velo_2_cam
            
            if self.load_pose:
                poses = self.parse_poses(osp.join(sequence_path, 'poses.txt'))

            glob_path = osp.join(sequence_path, 'voxels', '*.bin')

            for voxel_path in sorted(glob.glob(glob_path)):
                self.scans.append({
                    'sequence': sequence,
                    'P': P,
                    'T_velo_2_cam': T_velo_2_cam,
                    'proj_matrix': proj_matrix,
                    'voxel_path': voxel_path,
                })
                if self.load_pose:
                    frame_id = osp.splitext(osp.basename(voxel_path))[0]
                    self.scans[-1]['pose'] = poses[int(frame_id)]

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = self.scans[idx]
        sequence = scan['sequence']
        P = scan['P']
        T_velo_2_cam = scan['T_velo_2_cam']
        proj_matrix = scan['proj_matrix']

        filename = osp.basename(scan['voxel_path'])
        frame_id = osp.splitext(filename)[0]
        data = {
            'frame_id': frame_id,
            'sequence': sequence,
            'P': P,
            'cam_pose': T_velo_2_cam,
            'proj_matrix': proj_matrix,
            'voxel_origin': self.voxel_origin
        }
        label = {}

        scale_3ds = (self.output_scale, self.project_scale)
        data['scale_3ds'] = scale_3ds
        cam_K = P[:3, :3]
        data['cam_K'] = cam_K
        
        vanishing_point_path = osp.join(
            self.data_root, "vanishing_points", "seq_"+sequence+".json"
        )

        with open(vanishing_point_path, 'r') as v:
            vanishing_point_total = json.load(v)
            
        if len(sequence) > 5:
            #corner_case
            base_dict = 'SemanticKITTI/dataset/sequences/'+sequence[:2]+'/image_2'
        else:
            base_dict = 'SemanticKITTI/dataset/sequences/'+sequence+'/image_2'
            
        vanishing_point = np.array(vanishing_point_total[osp.join(base_dict, frame_id + ".png")][0], dtype=float)
        
        for scale_3d in scale_3ds:
            projected_pix, fov_mask, pix_z = vox2pix(T_velo_2_cam, cam_K, self.voxel_origin,
                                                     self.voxel_size * scale_3d, self.img_shape,
                                                     self.scene_size)
            data[f'projected_pix_{scale_3d}'] = projected_pix
            data[f'pix_z_{scale_3d}'] = pix_z
            data[f'fov_mask_{scale_3d}'] = fov_mask

        flip = random.random() > 0.5 if self.flip and self.split == 'train' else False
                
        target_1_path = osp.join(self.label_root, sequence, frame_id + '_1_1.npy')
        
        with_target = self.split != 'test' and osp.exists(target_1_path)
        
        if with_target:
            target = np.load(target_1_path)

        if self.depth_root is not None:
            depth_path = osp.join(self.depth_root, 'sequences', sequence, frame_id + '.npy')
            depth = np.load(depth_path)[:self.img_shape[1], :self.img_shape[0]]

        if self.load_pose:
            data['pose'] = scan['pose']

        img_path = osp.join(self.data_root, 'dataset', 'sequences', sequence, 'image_2',
                            frame_id + '.png')
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img[:self.img_shape[1], :self.img_shape[0]]
        
        if flip:
            target = np.flip(target, axis=1).copy()
            vanishing_point[0] = self.img_shape[0] - vanishing_point[0]
            depth = np.flip(depth, axis=1).copy()
            img = np.flip(img, axis=1).copy()
        
        if not self.split == 'test':
            label['target'] = target
        data['v_pts'] = vanishing_point
        data['depth'] = depth
        data['img'] = self.transforms(img)

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)
        return data, label

    @staticmethod
    def read_calib(calib_path):
        calib_data = {}
        with open(calib_path) as f:
            for line in f:
                if line == '\n':
                    break
                key, value = line.strip().split(':', 1)
                calib_data[key] = np.array([float(v) for v in value.split()])

        ret = {}
        ret['P2'] = calib_data['P2'].reshape(3, 4)
        ret['Tr'] = np.identity(4)
        ret['Tr'][:3, :4] = calib_data['Tr'].reshape(3, 4)
        return ret

    def parse_poses(self, filename):
        poses = []
        with open(filename) as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                pose = np.zeros((4, 4))
                pose[:3] = np.array(values).reshape((3, 4))
                pose[3, 3] = 1.0
                poses.append(pose)
        return poses