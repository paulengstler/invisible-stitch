import os
import sys
import torch
import pytorch3d
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from enum import Enum
from typing import Optional

# taken from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/shader.py

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds
from typing import cast
class PointsRendererWithMasks(PointsRenderer):
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists
        weights = 1 - dists2 / (r * r)
        ok = cast(torch.BoolTensor, (fragments.idx >= 0)).float()

        weights = weights * ok

        fragments_prm = fragments.idx.long().permute(0, 3, 1, 2)
        weights_prm = weights.permute(0, 3, 1, 2)
        images = self.compositor(
            fragments_prm,
            weights_prm,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        cumprod = torch.cumprod(1 - weights, dim=-1)
        cumprod = torch.cat((torch.ones_like(cumprod[..., :1]), cumprod[..., :-1]), dim=-1)
        depths = (weights * cumprod * fragments.zbuf).sum(dim=-1)

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        masks = fragments.idx.long()[..., 0] >= 0

        return images, masks, depths

def render(cameras, point_cloud, image_size):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = 1e-2,
        points_per_pixel = 10
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRendererWithMasks(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    # Render the scene
    return renderer(point_cloud)

def load_hypersim_camera_pose(camera_dir, frame_id, device="cpu"):
    camera_positions_hdf5_file    = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
    camera_orientations_hdf5_file = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")

    with h5py.File(camera_positions_hdf5_file,    "r") as f: camera_positions    = f["dataset"][:]
    with h5py.File(camera_orientations_hdf5_file, "r") as f: camera_orientations = f["dataset"][:]

    camera_position_world = camera_positions[frame_id]
    R_world_from_cam      = camera_orientations[frame_id]

    t_world_from_cam = np.matrix(camera_position_world).T
    R_cam_from_world = np.matrix(R_world_from_cam).T
    t_cam_from_world = -R_cam_from_world*t_world_from_cam

    M_cam_from_world = np.matrix(np.block([[R_cam_from_world,       t_cam_from_world],
                                           [np.matrix(np.zeros(3)), 1.0]]))

    M_p3dcam_from_cam = np.matrix(np.identity(4))
    M_p3dcam_from_cam[0,0] = -1
    M_p3dcam_from_cam[2,2] = -1

    M_p3dcam_from_world = M_p3dcam_from_cam * M_cam_from_world

    R_cam_from_world = M_p3dcam_from_world[:3,:3].T
    t_cam_from_world = M_p3dcam_from_world[:3,3]

    R_cam_from_world = torch.tensor(R_cam_from_world, dtype=torch.float32, device=device)
    t_cam_from_world = torch.tensor(t_cam_from_world, dtype=torch.float32, device=device)

    return R_cam_from_world, t_cam_from_world

def load_hypersim_image(img_dir, frame_id, device="cpu"):
    color_hdf5_file = os.path.join(img_dir, f"frame.{frame_id:04d}.color.hdf5")

    with h5py.File(color_hdf5_file, "r") as f: color = f["dataset"][:]

    color = torch.tensor(color, dtype=torch.float32, device=device)

    return color

def load_hypersim_point_cloud(geometry_dir, final_dir, frame_id, device="cpu"):
    point_positions_hdf5_file = os.path.join(geometry_dir, f"frame.{frame_id:04d}.position.hdf5")
    color_hdf5_file           = os.path.join(final_dir, f"frame.{frame_id:04d}.color.hdf5")

    with h5py.File(point_positions_hdf5_file, "r") as f: point_positions = f["dataset"][:]
    with h5py.File(color_hdf5_file, "r") as f: color = f["dataset"][:]

    point_positions = torch.tensor(point_positions, dtype=torch.float32, device=device)
    color = torch.tensor(color, dtype=torch.float32, device=device)

    point_cloud = Pointclouds(points=point_positions.reshape(1, -1, 3), features=color.reshape(1, -1, 3))

    return point_cloud

def load_hypersim_depth(geometry_dir, frame_id, device="cpu"):
    depth_hdf5_file = os.path.join(geometry_dir, f"frame.{frame_id:04d}.depth_meters.hdf5")
    with h5py.File(depth_hdf5_file, "r") as f: npyDistance = f["dataset"][:]

    # https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
    intWidth, intHeight = npyDistance.shape[1], npyDistance.shape[0]
    fltFocal = 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal

    depth = torch.tensor(npyDepth, dtype=torch.float32, device=device)
    return depth

def find_hypersim_pairs(hypersim_dir, scene, metadata_camera_parameters_csv_file):
    device = torch.device("cuda:0")

    import os

    from tqdm.auto import tqdm

    df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")

    scene_dir = os.path.join(hypersim_dir, scene)
    metadata_camera_csv_file = os.path.join(scene_dir, "_detail", "metadata_cameras.csv")

    scene_pairs = []

    df_ = df_camera_parameters.loc[scene]

    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])

    fov_x = df_["settings_camera_fov"]
    fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x/2.0) / width_pixels)

    camera_metadata = pd.read_csv(metadata_camera_csv_file)

    metadata_scene_csv_file = os.path.join(scene_dir, "_detail", "metadata_scene.csv")
    scene_metadata = pd.read_csv(metadata_scene_csv_file)

    meters_per_asset_unit = scene_metadata[scene_metadata["parameter_name"] == "meters_per_asset_unit"]["parameter_value"].values[0]

    for camera_name in camera_metadata["camera_name"].tolist():
        camera_dir = os.path.join(scene_dir, "_detail", camera_name)
        img_dir = os.path.join(scene_dir, "images")
        final_dir = os.path.join(img_dir, f"scene_{camera_name}_final_hdf5")
        geometry_dir = os.path.join(img_dir, f"scene_{camera_name}_geometry_hdf5")

        with h5py.File(os.path.join(camera_dir, "camera_keyframe_frame_indices.hdf5"), "r") as f: frame_indices = list(f["dataset"][:])

        camera_poses = [load_hypersim_camera_pose(camera_dir, frame_id, "cpu") for frame_id in frame_indices]

        pbar = tqdm(frame_indices)
        for frame_id in pbar:
            remaining_frame_ids = frame_indices[frame_indices.index(frame_id)+1:]
            if len(remaining_frame_ids) == 0: break

            try:
                point_cloud = load_hypersim_point_cloud(geometry_dir, final_dir, frame_id, device)
            except FileNotFoundError:
                continue

            for remaining_frame_id in remaining_frame_ids:
                R_cam_from_world, t_cam_from_world = camera_poses[remaining_frame_id]

                cameras = FoVPerspectiveCameras(
                    R=R_cam_from_world.view(1, 3, 3), 
                    T=t_cam_from_world.view(1, 3), 
                    znear=0.1, 
                    zfar=400.0, 
                    aspect_ratio=1.0, 
                    fov=fov_y,
                    degrees=False,
                    device=device
                )
                _, masks, depths = render(cameras, point_cloud, (height_pixels, width_pixels))

                overlap = masks[0].sum() / (height_pixels * width_pixels)

                pbar.set_description(f"{scene}_{camera_name}, {frame_id}-{remaining_frame_id} ({len(scene_pairs)})")

                # these frames don't overlap sufficiently
                if overlap.item() < 0.8: continue

                try:
                    gt_depth = load_hypersim_depth(geometry_dir, remaining_frame_id, device)
                    gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
                except:
                    continue

                sparse_gt_error = torch.mean(torch.abs(gt_depth - depths[0]*meters_per_asset_unit)[depths[0] > 0])

                # there's a major reprojection error, likely due to occlusion
                if sparse_gt_error > 0.1: continue

                scene_pairs.append((scene, camera_name, frame_id, remaining_frame_id, overlap.item()))

    # write pairs to csv
    df_pairs = pd.DataFrame(scene_pairs, columns=["scene", "camera", "frame_id", "remaining_frame_id", "overlap"])
    df_pairs.to_csv(os.path.join(hypersim_dir, f"{scene}_pairs.csv"))

def run_jobs(hypersim_dir, scene_idx="all", metadata_camera_parameters_csv_file="./datasets/hypersim/metadata_camera_parameters.csv"):
    df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")

    scenes = df_camera_parameters.index.tolist()
    scenes = [(scene if not df_camera_parameters.loc[scene]["use_camera_physical"] else None) for scene in scenes]

    if scene_idx != "all":
        my_scene = scenes[scene_idx]
        if my_scene is not None:
            find_hypersim_pairs(hypersim_dir, my_scene, metadata_camera_parameters_csv_file)

    else:
        for i in range(0, 100):
            my_scene = scenes[i]
            if my_scene is not None:
                find_hypersim_pairs(hypersim_dir, my_scene, metadata_camera_parameters_csv_file)


if __name__ == "__main__":
    import fire
    fire.Fire(run_jobs)
    