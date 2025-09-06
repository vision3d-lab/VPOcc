import os
import pickle

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.progress import track
from mayavi import mlab

# Predefined RGBA color maps for SemanticKITTI and KITTI-360
COLORS = np.array([
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255],
]).astype(np.uint8)

KITTI360_COLORS = np.concatenate((
    COLORS[0:6],
    COLORS[8:15],
    COLORS[16:],
    np.array([[250, 150, 0, 255], [50, 255, 255, 255]]).astype(np.uint8),
), 0)



def get_grid_coords(dims, resolution):
    """
    Generate voxel grid center coordinates.
    dims: grid size [x, y, z]
    resolution: voxel size
    """
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(float)
    coords_grid = (coords_grid * resolution) + resolution / 2

    # Swap x, y axes for visualization
    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    return temp


def draw(
    voxels,
    cam_pose,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # camera mesh depth
    colors=None,
    save=False,
    save_mode='auto',
):
    """
    Visualize predicted/target voxel grids with camera frustum.
    - voxels: occupancy grid
    - cam_pose: camera extrinsics
    - vox_origin: voxel grid origin
    - fov_mask: field-of-view mask
    """
    # Compute camera frustum points
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([[0,0,0],[x,y,d],[-x,y,d],[-x,-y,d],[x,-y,d]])
    tri_points = (np.linalg.inv(cam_pose) @ np.hstack([tri_points, np.ones((5,1))]).T).T
    x, y, z = tri_points[:,0]-vox_origin[0], tri_points[:,1]-vox_origin[1], tri_points[:,2]-vox_origin[2]
    triangles = [(0,1,2),(0,1,4),(0,3,4),(0,2,3)]

    # Prepare grid coords and mask
    grid_coords = get_grid_coords(voxels.shape, voxel_size)
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    fov_grid_coords = grid_coords[fov_mask, :]
    outfov_grid_coords = grid_coords[~fov_mask, :]

    mlab.figure(bgcolor=(1, 1, 1), size=(1280, 1280))
    mlab.triangular_mesh(x, y, z, triangles, representation='wireframe', color=(0, 0, 0), line_width=4)

    # Adjust colors for out-of-FOV voxels
    outfov_colors = colors.copy()
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2

    # Draw FOV and out-of-FOV voxels
    for i, grid_coords in enumerate((fov_grid_coords, outfov_grid_coords)):
        voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]
        plt_plot = mlab.points3d(
            voxels[:, 0], voxels[:, 1], voxels[:, 2], voxels[:, 3],
            colormap='viridis',
            scale_factor=voxel_size * 0.95,
            mode='cube',
            opacity=1.0,
            vmin=1,
            vmax=19)
        plt_plot.glyph.scale_mode = 'scale_by_vector'
        plt_plot.module_manager.scalar_lut_manager.lut.table = colors if i == 0 else outfov_colors

    # Camera view and save
    mlab.view(azimuth=225, elevation=50)
    plt_plot.scene.camera.zoom(0.7)
    mlab.savefig(save, size=(520, 520))

    if save_mode == 'auto':
        mlab.close()
    elif save_mode == 'manual':
        mlab.show()
    else:
        raise ValueError(f"Invalid save_mode: {save_mode}")


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(config: DictConfig):
    """
    Main visualization loop.
    Loads prediction files and renders both prediction and ground truth.
    """
    # Collect files
    files = ([os.path.join(config.path, f) for f in os.listdir(config.path)]
             if os.path.isdir(config.path) else [config.path])
    output_dir, save_mode = config.output_dir, config.save_mode
    os.makedirs(output_dir, exist_ok=True)

    for file in track(files):
        with open(file, 'rb') as f:
            outputs = pickle.load(f)

        cam_pose = outputs.get('cam_pose', outputs['T_velo_2_cam'])
        fov_mask, target = outputs['fov_mask_1'], outputs['target']
        vox_origin = np.array([0, -25.6, -2])

        # Handle prediction field names
        try:
            pred = outputs.get('pred', outputs['y_pred'])
            if pred.shape != (256, 256, 32):
                pred = pred.reshape(256, 256, 32)
        except:
            pred = outputs['output_voxel'].reshape(256, 256, 32)

        # Dataset-specific params
        if config.data.datasets.type == 'SemanticKITTI':
            params = dict(img_size=(1220, 370), f=707.0912, voxel_size=0.2, d=7, colors=COLORS)
        elif config.data.datasets.type == 'KITTI360':
            pred[target == 255] = 0  # Ignore label correction
            params = dict(img_size=(1408, 376), f=552.55426, voxel_size=0.2, d=7, colors=KITTI360_COLORS)
        else:
            raise NotImplementedError

        file_name = os.path.basename(file).split(".")[0]

        # Save both prediction and target visualizations
        draw(pred, cam_pose, vox_origin, fov_mask, **params,
            save=os.path.join(output_dir, f'{file_name}_pred.png'), save_mode=save_mode)
        draw(target, cam_pose, vox_origin, fov_mask, **params,
            save=os.path.join(output_dir, f'{file_name}_target.png'), save_mode=save_mode)


if __name__ == '__main__':
    main()