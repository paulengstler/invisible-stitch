import re
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
import cv2
from scipy import ndimage

import skimage
from PIL import Image

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

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

def project_points(cameras, depth, focal_length, principal_point, use_pixel_centers=True):
    # only works with a single camera for now
    depth_t = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth
    depth_t = depth_t.to(cameras.device)

    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = focal_length
    cx, cy = principal_point

    # assume all cameras render images of the same size
    i, j = torch.meshgrid(
        torch.arange(cameras.image_size[0][1], dtype=torch.float32, device=cameras.device) + pixel_center,
        torch.arange(cameras.image_size[0][0], dtype=torch.float32, device=cameras.device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth_t / fx, -(j - cy) * depth_t / fy, depth_t], -1
    )

    xy_depth_world = cameras.get_world_to_view_transform().inverse().transform_points(directions.view(-1, 3)).unsqueeze(0)

    return xy_depth_world

def get_pointcloud(xy_depth_world, features=None):
    point_cloud = Pointclouds(points=[xy_depth_world], features=[features] if features is not None else None)
    return point_cloud

def merge_pointclouds(point_clouds):
    points = torch.cat([pc.points_padded() for pc in point_clouds], dim=1)
    features = torch.cat([pc.features_padded() for pc in point_clouds], dim=1)
    return Pointclouds(points=[points[0]], features=[features[0]])

def render(cameras, point_cloud, **kwargs):
    raster_settings = PointsRasterizationSettings(
        image_size=(int(cameras.image_size[0, 0]), int(cameras.image_size[0, 1])), 
        radius = 1e-2,
        points_per_pixel = 16
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRendererWithMasks(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    # Render the scene
    return renderer(point_cloud, **kwargs)

def align_depths_in_world_with_offset(cameras, previous_depth, new_depth, focal_length, principal_point,tolerance = 1e-5):
    depth_scalar = torch.tensor([1.0], requires_grad=True, device=new_depth.device)
    depth_offset = torch.rand(1, requires_grad=True, device=new_depth.device)

    depth_mask = (previous_depth.view(-1) > 0)

    original_points = project_points(cameras, previous_depth, focal_length, principal_point)[0]
    masked_original_points = original_points[depth_mask]

    optimizer = torch.optim.Adam((depth_scalar, depth_offset), lr=1e-2)
    previous_loss = torch.inf
    grace_period = 10
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(project_points(cameras, depth_scalar * new_depth + depth_offset, focal_length, principal_point)[0][depth_mask], masked_original_points, reduction="mean")

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

def nearest_neighbor_fill(img, mask):
    img_ = np.copy(img.cpu().numpy())

    # we need to slightly erode the mask to avoid interacting with upsampling artifacts
    eroded_mask = skimage.morphology.binary_erosion(mask.cpu().numpy(), footprint=skimage.morphology.disk(3))

    img_[eroded_mask <= 0] = np.nan

    distance_to_boundary = ndimage.distance_transform_bf((~eroded_mask>0), metric="cityblock")

    for current_dist in np.unique(distance_to_boundary)[1:]:
        ii, jj = np.where(distance_to_boundary == current_dist)
        
        # get 3x3 neighborhood
        ii_ = np.array([ii - 1, ii, ii + 1, ii - 1, ii, ii + 1, ii - 1, ii, ii + 1]).reshape(9, -1)
        jj_ = np.array([jj - 1, jj - 1, jj - 1, jj, jj, jj, jj + 1, jj + 1, jj + 1]).reshape(9, -1)

        ii_ = ii_.clip(0, img_.shape[0] - 1)
        jj_ = jj_.clip(0, img_.shape[1] - 1)

        # get the mean of the neighborhood
        img_[ii, jj] = np.nanmax(img_[ii_, jj_], axis=0)

    return torch.from_numpy(img_).to(img.device)

@torch.no_grad()
def infer_with_zoe_dc(zoe_dc, image, sparse_depth):
    from torchvision import transforms

    if not hasattr(zoe_dc, "vanilla") or not zoe_dc.vanilla:
        sparse_depth_mask = (sparse_depth[None, None, ...] > 0).float()
        x = torch.cat([image[None, ...], sparse_depth[None, None, ...] / 10.0, sparse_depth_mask], dim=1).to(zoe_dc.device)

        pred_depth = zoe_dc(x)["metric_depth"]
    else:
        pred_depth = zoe_dc(image[None, ...].to(zoe_dc.device))["metric_depth"]

    return torch.nn.functional.interpolate(pred_depth, image.shape[-2:], mode='bilinear', align_corners=True)[0, 0].to(sparse_depth.device)

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

def get_dc_zoe_model(cktp_path, vanilla=False, **kwargs):
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

        config.__setattr__("checkpoint", ckpt_path)

        model = load_ckpt(config, model)

        return model

    return ZoeD_N(ckpt_path, vanilla=vanilla, **kwargs)

from typing import Literal, Optional
def run_scannet_scene(
        scene: str = "scene0000_00",
        zoe_dc_model = None,
        ckpt_path: Optional[str] = None,
        mode: Literal["zero", "pcd"] = "pcd",
        align: bool = False,
        vanilla_zoe: bool = False,
        scene_seq_length: int = 2,
        device = None,
        exit_after_visualization: bool = False,
        start_from_gt: bool = True,
        compute_error_after_each_frame: bool = False,
    ):

    assert zoe_dc_model is None or ckpt_path is None, "Either provide a model or a checkpoint path, not both."

    # print all arguments
    if __name__ == "__main__":
        print(", ".join([f"{k}={v}" for k, v in locals().items()]))

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    if zoe_dc_model is None:
        zoe_dc_model = get_dc_zoe_model(vanilla=vanilla_zoe, ckpt_path=ckpt_path).to(device)

    if mode == "zero" and not align:
        print("Running ZoeDepth in zero mode without aligning the depths.")
        import time
        time.sleep(5)
    
    import os
    scene_base_dir = os.path.join("/scratch/shared/nfs2/paule/ScanNet/", scene)
    scene_rgb_dir = os.path.join(scene_base_dir, "color")
    scene_depth_dir = os.path.join(scene_base_dir, "depth")
    scene_pose_dir = os.path.join(scene_base_dir, "pose")

    scene_pose_files = natural_sort(os.listdir(scene_pose_dir))
    scene_pose_files = [os.path.join(scene_pose_dir, f) for f in scene_pose_files]

    def lazy_load_scene_pose(index):
        return np.loadtxt(scene_pose_files[index])

    if __name__ == "__main__":
        print(f"Found {len(scene_pose_files)} poses")

    intrinsics_file = os.path.join(scene_base_dir, "intrinsic", "intrinsic_depth.txt")
    intrinsics = np.loadtxt(intrinsics_file)

    from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection, opencv_from_cameras_projection
    from tqdm.auto import tqdm

    visualized_scenes = 0
    absolute_scene_errors = []
    pbar = tqdm(range(0, len(scene_pose_files), 50), total=len(scene_pose_files) // 50)
    for partial_scene_start in pbar:
        global point_cloud
        point_cloud = None

        offset = partial_scene_start
        for i in range(scene_seq_length):
            frame_i = (i * 10) + offset

            if frame_i >= len(scene_pose_files):
                break

            scene_rgb_file = os.path.join(scene_rgb_dir, scene_pose_files[frame_i].split('/')[-1].split('.')[0] + '.jpg')
            scene_rgb = np.array(Image.open(scene_rgb_file))

            scene_depth_file = os.path.join(scene_depth_dir, scene_pose_files[frame_i].split('/')[-1].split('.')[0] + '.png')
            scene_depth = np.array(Image.open(scene_depth_file)).astype(np.int32)

            scene_rgb = cv2.resize(scene_rgb, (scene_depth.shape[1], scene_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

            R, T = torch.from_numpy(lazy_load_scene_pose(frame_i)[:3, :3]).float(), torch.from_numpy(lazy_load_scene_pose(frame_i)[:3, 3]).float()
            pose = torch.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = T

            extrinsics = torch.linalg.inv(pose)
            R, T = extrinsics[:3, :3], extrinsics[:3, 3]

            K = torch.from_numpy(intrinsics).unsqueeze(0)
            image_size = torch.tensor([480, 640]).view(1, 2)

            global focal_length, principal_point

            focal_length = K[0, :2, :2].diagonal()
            principal_point = K[0, :2, 2]

            global cameras
            cameras = cameras_from_opencv_projection(R.view(1, 3, 3), T.view(1, 3), K[:, :3, :3].view(1, 3, 3).float(), image_size)
            cameras.image_size = image_size
            cameras = cameras.to(device)

            if start_from_gt and i == 0:
                try:
                    projected_points = project_points(cameras, torch.from_numpy(scene_depth).float() / 1000., focal_length, principal_point)
                except:
                    break
    
                projection_mask = torch.from_numpy(scene_depth).float().view(-1) > 0
            elif not start_from_gt and i == 0:
                estimated_depth = infer_with_zoe_dc(zoe_dc_model, torch.from_numpy(scene_rgb).float().permute(2, 0, 1) / 255., torch.zeros_like(torch.from_numpy(scene_depth)).float())

                try:
                    projected_points = project_points(cameras, estimated_depth, focal_length, principal_point)
                except:
                    break
                    
                projection_mask = torch.from_numpy(scene_depth).float().view(-1) > 0
            else:
                images, masks, depths = render(cameras, point_cloud)

                # rgb values == 0 do not seem to be sufficient to mask out the border
                projection_mask = ~masks[0] & torch.from_numpy((scene_rgb > 15).all(-1)).to(masks.device)

                if mode == "zero":
                    estimated_depth = infer_with_zoe_dc(zoe_dc_model, torch.from_numpy(scene_rgb).float().permute(2, 0, 1) / 255., torch.zeros_like(torch.from_numpy(scene_depth)).float())
        
                elif mode == "pcd":
                    # slightly erode the mask to avoid aliasing artifacts
                    eroded_mask = skimage.morphology.binary_erosion((depths[0] > 0).cpu().numpy(), footprint=None)
                    eroded_depth = depths[0].clone()
                    eroded_depth[torch.from_numpy(eroded_mask).to(depths.device) <= 0] = 0

                    estimated_depth = infer_with_zoe_dc(zoe_dc_model, torch.from_numpy(scene_rgb).float().permute(2, 0, 1) / 255., eroded_depth.cpu())

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                if align:
                    try:
                        estimated_depth = align_depths_in_world_with_offset(cameras, depths[0], estimated_depth, focal_length, principal_point)
                    except:
                        break

                try:
                    projected_points = project_points(cameras, estimated_depth, focal_length, principal_point)
                except:
                    break

            rgb = (torch.from_numpy(np.asarray(scene_rgb).copy()).reshape(-1, 3).float() / 255).to(device)

            if i == 0:
                point_cloud = get_pointcloud(projected_points[0][projection_mask.view(-1)], rgb[projection_mask.view(-1)])
            else:
                if not projection_mask.any(): continue
                point_cloud = merge_pointclouds([point_cloud, get_pointcloud(projected_points[0][projection_mask.view(-1)], rgb[projection_mask.view(-1)])])

            # calculate absolute difference between estimated and gt depth
            if compute_error_after_each_frame or i == scene_seq_length - 1:
                #_, _, depths = render(cameras, point_cloud)

                depth_error = torch.abs(estimated_depth.cpu() - torch.from_numpy(scene_depth).float() / 1000.)
                depth_error[scene_depth == 0] = 0
                absolute_scene_errors.append(depth_error[projection_mask.cpu()].mean())

                pbar.set_description(f"{torch.tensor(absolute_scene_errors).mean().item():.4f}")

    return absolute_scene_errors

def multi_scene_runner(
    scannet_path: str = "./datasets/ScanNet",
    scenes: int = 50,
    vanilla_zoe: bool = False,
    mode: Literal["zero", "pcd"] = "pcd",
    align: bool = False,
    scene_seq_length: int = 2,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="paulengstler/invisible-stitch", filename="invisible-stitch.pt")

    zoe_dc_model = get_dc_zoe_model(vanilla=vanilla_zoe, ckpt_path=ckpt_path).to(device)

    scannet_scenes = os.listdir(scannet_path)

    filtered_scenes = [s for s in scannet_scenes if any([f"scene{i:04d}" in s for i in range(scenes)])]

    print(f"Running {len(filtered_scenes)} scenes")

    scene_errors = []
    from tqdm.auto import tqdm
    for scene_name in tqdm(filtered_scenes):
        scene_errors.append(run_scannet_scene(
            scene=scene_name,
            zoe_dc_model=zoe_dc_model,
            mode=mode,
            align=align,
            scene_seq_length=scene_seq_length,
            vanilla_zoe=vanilla_zoe
        ))

    scene_errors = [item for sublist in scene_errors for item in sublist]

    print(f"Mean error: {torch.tensor(scene_errors).mean().item():.4f}")

if __name__ == "__main__":
    import fire
    fire.Fire(multi_scene_runner)
