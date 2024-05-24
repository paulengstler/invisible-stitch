import os
import sys
import torch
import pytorch3d

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.linalg
from enum import Enum
from typing import Optional
import skimage

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from pytorch3d.structures import Pointclouds
from typing import cast
class PointsRendererWithMasks(PointsRenderer):
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
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
        points_per_pixel = 16
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRendererWithMasks(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    # Render the scene
    return renderer(point_cloud)

try:
    from zoedepth.utils.misc import colorize
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model

except ImportError:
    import sys
    sys.path.insert(0, "..")

    from zoedepth.utils.misc import colorize
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model

def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    import glob
    import os

    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

def get_dc_zoe_model(ckpt_path, vanilla=False, **kwargs):
    def ZoeD_N(midas_model_type="DPT_BEiT_L_384", vanilla=False, **kwargs):
        if midas_model_type != "DPT_BEiT_L_384":
            raise ValueError(f"Only DPT_BEiT_L_384 MiDaS model is supported for pretrained Zoe_N model, got: {midas_model_type}")

        config = get_config("zoedepth", "train", **kwargs)
        model = build_model(config)

        if vanilla:
            model.__setattr__("vanilla", True)
            return model
        else:
            model.__setattr__("vanilla", False)

        if config.add_depth_channel:
            model.core.core.pretrained.model.patch_embed.proj = torch.nn.Conv2d(
                model.core.core.pretrained.model.patch_embed.proj.in_channels+2,
                model.core.core.pretrained.model.patch_embed.proj.out_channels,
                kernel_size=model.core.core.pretrained.model.patch_embed.proj.kernel_size,
                stride=model.core.core.pretrained.model.patch_embed.proj.stride,
                padding=model.core.core.pretrained.model.patch_embed.proj.padding,
                bias=True)

        assert os.path.exists(ckpt_path)
        config.__setattr__("checkpoint", ckpt_path)

        model = load_ckpt(config, model)

        return model

    return ZoeD_N(ckpt_path, vanilla=vanilla, **kwargs)

@torch.no_grad()
def infer_with_zoe_dc(zoe_dc, image, sparse_depth):
    from torchvision import transforms

    if not hasattr(zoe_dc, "vanilla") or not zoe_dc.vanilla:
        sparse_depth_mask = (sparse_depth[None, None, ...] > 0).float()
        x = torch.cat([image[None, ...], sparse_depth[None, None, ...] / 10.0, sparse_depth_mask], dim=1).to(zoe_dc.device)

        # FIXME get rid off hardcoded max depth 10
        pred_depth = 1 * zoe_dc(x)["metric_depth"]
    else:
        pred_depth = zoe_dc(image[None, ...].to(zoe_dc.device))["metric_depth"]

    return torch.nn.functional.interpolate(pred_depth, image.shape[-2:], mode='bilinear', align_corners=True)[0, 0].to(sparse_depth.device)

def outpaint_with_depth_estimation(image, mask, previous_depth, h, w, pipe, zoe_dc, prompt, dilation_size=2):
    img_input = Image.fromarray((255*image[..., :3].cpu().numpy()).astype(np.uint8))

    img_mask = Image.fromarray((255*skimage.morphology.isotropic_dilation(((~mask).cpu().numpy()), radius=dilation_size)).astype(np.uint8))#footprint=skimage.morphology.disk(dilation_size)))

    out_image = pipe(prompt=prompt, image=img_input, mask_image=img_mask, height=h, width=w).images[0]
    out_depth = infer_with_zoe_dc(zoe_dc, torch.from_numpy(np.asarray(out_image)/255.).permute(2,0,1).float().to(zoe_dc.device), (previous_depth * mask).to(zoe_dc.device)).cpu().numpy()

    return out_image, out_depth

def project_points(cameras, depth, image_size, focal_length, principal_point, use_pixel_centers=True):
    depth_t = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth
    depth_t = depth_t.to(cameras.device)

    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = focal_length
    cx, cy = principal_point

    i, j = torch.meshgrid(
        torch.arange(image_size[1], dtype=torch.float32, device=cameras.device) + pixel_center,
        torch.arange(image_size[0], dtype=torch.float32, device=cameras.device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth_t / fx, -(j - cy) * depth_t / fy, depth_t], -1
    )

    directions_hom = torch.cat((directions.view(-1, 3), torch.ones_like(directions.view(-1, 3)[:, :1])), dim=1)
    xy_depth_world_hom = (directions_hom.to(cameras.device) @ cameras.get_world_to_view_transform().inverse().get_matrix())[0]
    xy_depth_world = xy_depth_world_hom[:, :3] / xy_depth_world_hom[:, 3:]
    xy_depth_world = (xy_depth_world.view(-1, 3)).unsqueeze(0)

    return xy_depth_world.to(cameras.device)

def align_depths_in_world_with_offset(cameras, previous_depth, new_depth, image_size, focal_length, principal_point,tolerance = 1e-5):
    depth_scalar = torch.tensor([1.0], requires_grad=True, device=new_depth.device)
    depth_offset = torch.rand(1, requires_grad=True, device=new_depth.device)

    depth_mask = (previous_depth.view(-1) > 0)

    original_points = project_points(cameras, previous_depth, image_size, focal_length, principal_point)[0]
    masked_original_points = original_points[depth_mask]

    optimizer = torch.optim.Adam((depth_scalar, depth_offset), lr=1e-2)
    previous_loss = torch.inf
    grace_period = 10
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(project_points(cameras, depth_scalar * new_depth + depth_offset, image_size, focal_length, principal_point)[0][depth_mask], masked_original_points, reduction="mean")

        if previous_loss - loss < tolerance:
            if grace_period == 0:
                break
            else:
                grace_period -= 1
        else:
            grace_period = 10
        
        previous_loss = loss.item()

        loss.backward()
        optimizer.step()

    return (depth_scalar * new_depth + depth_offset).detach()

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

def load_hypersim_image(final_dir, frame_id, device="cpu"):
    color_hdf5_file = os.path.join(final_dir, f"frame.{frame_id:04d}.color.hdf5")
    with h5py.File(color_hdf5_file, "r") as f: color = f["dataset"][:]

    color = torch.tensor(color, dtype=torch.float32, device=device)

    return color

def load_hypersim_point_cloud(geometry_dir, final_dir, frame_id, device="cpu"):
    point_positions_hdf5_file = os.path.join(geometry_dir, f"frame.{frame_id:04d}.position.hdf5")
    color_hdf5_file           = os.path.join(final_dir, f"frame.{frame_id:04d}.color.hdf5")

    with h5py.File(point_positions_hdf5_file, "r") as f: point_positions = f["dataset"][:]
    with h5py.File(color_hdf5_file, "r") as f: color = f["dataset"][:]

    point_positions = torch.tensor(point_positions, dtype=torch.float32, device=device)
    color = torch.tensor(color, dtype=torch.float32, device=device).clip(0, 1)

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

def load_hypersim_image(img_dir, frame_id, device="cpu"):
    color_hdf5_file = os.path.join(img_dir, f"frame.{frame_id:04d}.color.hdf5")

    with h5py.File(color_hdf5_file, "r") as f: color = f["dataset"][:]

    color = torch.tensor(color, dtype=torch.float32, device=device)

    return color

import math
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def evaluate_scene(hypersim_dir, out_dir, scene,metadata_camera_parameters_csv_file,  models = ("inpainted", "zero", "zero_aligned", "zoedepth", "zoedepth_aligned")):
    device = torch.device("cuda:0")

    if not os.path.exists(os.path.join(hypersim_dir, f"{scene}_pairs.csv")):
        print(f"Pulling back, scene {scene} does not have a pairs csv file")
        return

    # get pairs csv file
    pairs = pd.read_csv(os.path.join(hypersim_dir, f"{scene}_pairs.csv"))

    if any(m in models for m in ("inpainted", "zero", "zero_aligned")):
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id="paulengstler/invisible-stitch", filename="invisible-stitch.pt")

        zoe_dc_model = get_dc_zoe_model(ckpt_path).to(device)

    if any(m in models for m in ("zoedepth", "zoedepth_aligned")):
        zoe_vanilla_model = get_dc_zoe_model(".", vanilla=True).to(device)

    from tqdm.auto import tqdm

    df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")

    scene_dir = os.path.join(hypersim_dir, scene)
    metadata_camera_csv_file = os.path.join(scene_dir, "_detail", "metadata_cameras.csv")

    df_ = df_camera_parameters.loc[scene]

    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])

    fov_x = df_["settings_camera_fov"]
    fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x/2.0) / width_pixels)

    focal_y = fov2focal(fov_y, height_pixels)

    camera_metadata = pd.read_csv(metadata_camera_csv_file)

    metadata_scene_csv_file = os.path.join(scene_dir, "_detail", "metadata_scene.csv")
    scene_metadata = pd.read_csv(metadata_scene_csv_file)

    meters_per_asset_unit = scene_metadata[scene_metadata["parameter_name"] == "meters_per_asset_unit"]["parameter_value"].values[0]

    errors = {
        k: [] for k in models
    }

    for camera_name in camera_metadata["camera_name"].tolist():
        camera_dir = os.path.join(scene_dir, "_detail", camera_name)
        img_dir = os.path.join(scene_dir, "images")
        final_dir = os.path.join(img_dir, f"scene_{camera_name}_final_hdf5")
        geometry_dir = os.path.join(img_dir, f"scene_{camera_name}_geometry_hdf5")

        # find pairs for this camera
        camera_pairs = pairs[pairs["camera"] == camera_name]

        # get all frame ids to prefetch the poses
        frame_ids = set(camera_pairs["frame_id"].tolist() + camera_pairs["remaining_frame_id"].tolist())

        with h5py.File(os.path.join(camera_dir, "camera_keyframe_frame_indices.hdf5"), "r") as f: frame_indices = f["dataset"][:]
        frame_count = frame_indices.shape[0]

        camera_poses = [load_hypersim_camera_pose(camera_dir, frame_id, "cpu") if frame_id in frame_ids else None for frame_id in range(frame_count)]

        pbar = tqdm(list(zip(camera_pairs["frame_id"], camera_pairs["remaining_frame_id"])))
        for (left_frame_id, right_frame_id) in pbar:
            point_cloud = load_hypersim_point_cloud(geometry_dir, final_dir, left_frame_id, device)

            R_cam_from_world, t_cam_from_world = camera_poses[right_frame_id]

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

            images, masks, depths = render(cameras, point_cloud, (height_pixels, width_pixels))

            eroded_mask = skimage.morphology.binary_erosion((depths[0] > 0).cpu().numpy(), footprint=None)
            eroded_depth = depths[0].clone()
            eroded_depth[torch.from_numpy(eroded_mask).to(depths.device) <= 0] = 0

            try:
                scene_rgb = load_hypersim_image(final_dir, right_frame_id, device).clip(0, 1)
            except FileNotFoundError:
                # ai_001_001/images/scene_cam_00_final_hdf5/frame.0061.color.hdf5 seems to be missing in the original data
                for k in errors.keys():
                    errors[k].append(0)

                continue

            gt_depth = load_hypersim_depth(geometry_dir, right_frame_id, device)
            gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

            evaluation_mask = ~masks[0]

            sparse_depth = eroded_depth*meters_per_asset_unit

            if "inpainted" in models:
                with torch.no_grad():
                    inpaint_estimated = infer_with_zoe_dc(zoe_dc_model, scene_rgb.permute(2, 0, 1), sparse_depth)

                errors["inpainted"].append(torch.mean(torch.abs(inpaint_estimated - gt_depth)[evaluation_mask]).item())

            if any(m in models for m in ("zero", "zero_aligned")):
                with torch.no_grad():
                    zero_estimated = infer_with_zoe_dc(zoe_dc_model, scene_rgb.permute(2, 0, 1), torch.zeros_like(eroded_depth*meters_per_asset_unit))

            if "zero" in models:   
                errors["zero"].append(torch.mean(torch.abs(zero_estimated - gt_depth)[evaluation_mask]).item())

            cameras.__setattr__("image_size", (height_pixels, width_pixels))

            if "zero_aligned" in models:
                zero_aligned_estimated = align_depths_in_world_with_offset(cameras, depths[0]*meters_per_asset_unit, zero_estimated, (height_pixels, width_pixels), (focal_y, focal_y), (width_pixels/2, height_pixels/2))
            
                errors["zero_aligned"].append(torch.mean(torch.abs(zero_aligned_estimated - gt_depth)[evaluation_mask]).item())

            if any(m in models for m in ("zoedepth", "zoedepth_aligned")):
                with torch.no_grad():
                    zoedepth_estimated = infer_with_zoe_dc(zoe_vanilla_model, scene_rgb.permute(2, 0, 1), sparse_depth)

            if "zoedepth" in models:
                errors["zoedepth"].append(torch.mean(torch.abs(zoedepth_estimated - gt_depth)[evaluation_mask]).item())

            if "zoedepth_aligned" in models:
                zoedepth_aligned_estimated = align_depths_in_world_with_offset(cameras, depths[0]*meters_per_asset_unit, zoedepth_estimated, (height_pixels, width_pixels), (focal_y, focal_y), (width_pixels/2, height_pixels/2))

                errors["zoedepth_aligned"].append(torch.mean(torch.abs(zoedepth_aligned_estimated - gt_depth)[evaluation_mask]).item())

    results = pd.DataFrame({
        k: v for k, v in errors.items()
    })

    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, f"{scene}_results.csv"))

def run_evaluation(out_dir, scene_idx = "all", hypersim_dir = "./datasets/hypersim", models = ("inpainted",), metadata_camera_parameters_csv_file="./datasets/hypersim/metadata_camera_parameters.csv"):
    df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")

    scenes = df_camera_parameters.index.tolist()
    scenes = [(scene if not df_camera_parameters.loc[scene]["use_camera_physical"] else None) for scene in scenes]

    if scene_idx != "all":
        my_scene = scenes[scene_idx]
        if my_scene is not None:
            evaluate_scene(hypersim_dir, out_dir, my_scene, metadata_camera_parameters_csv_file, models=models)

    else:
        for i in range(0, 100):
            my_scene = scenes[i]
            if my_scene is not None:
                evaluate_scene(hypersim_dir, out_dir, my_scene, metadata_camera_parameters_csv_file, models=models)

if __name__ == "__main__":
    import fire
    fire.Fire(run_evaluation)
