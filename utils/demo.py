import copy
import torch
import numpy as np

import skimage
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
)

from .render import render
from .ops import project_points, get_pointcloud, merge_pointclouds

def downsample_point_cloud(optimization_bundle, device="cpu"):
    point_cloud = None

    for i, frame in enumerate(optimization_bundle["frames"]):
        if frame.get("supporting", False):
            continue

        downsampled_image = copy.deepcopy(frame["image"])
        downsampled_image.thumbnail((360, 360))

        image_size = downsampled_image.size
        w, h = image_size

        # regenerate the point cloud at a lower resolution
        R, T = look_at_view_transform(device=device, azim=frame["azim"], elev=frame["elev"], dist=frame["dist"])#, dist=1+0.15*step)
        cameras = PerspectiveCameras(R=R, T=T, focal_length=torch.tensor([w], dtype=torch.float32), principal_point=(((h-1)/2, (w-1)/2),), image_size=(image_size,), device=device, in_ndc=False)

        # downsample the depth
        downsampled_depth = torch.nn.functional.interpolate(torch.tensor(frame["depth"]).unsqueeze(0).unsqueeze(0).float().to(device), size=(h, w), mode="nearest").squeeze()

        xy_depth_world = project_points(cameras, downsampled_depth)

        rgb = (torch.from_numpy(np.asarray(downsampled_image).copy()).reshape(-1, 3).float() / 255).to(device)

        c2w = cameras.get_world_to_view_transform().get_matrix()[0]

        if i == 0:
            point_cloud = get_pointcloud(xy_depth_world[0], device=device, features=rgb)

        else:
            images, masks, depths = render(cameras, point_cloud, radius=1e-2)

            # pytorch 3d's mask might be slightly too big (subpixels), so we erode it a little to avoid seams
            # in theory, 1 pixel is sufficient but we use 2 to be safe
            masks[0] = torch.from_numpy(skimage.morphology.binary_erosion(masks[0].cpu().numpy(), footprint=skimage.morphology.disk(1))).to(device)

            partial_outpainted_point_cloud = get_pointcloud(xy_depth_world[0][~masks[0].view(-1)], device=device, features=rgb[~masks[0].view(-1)])

            point_cloud = merge_pointclouds([point_cloud, partial_outpainted_point_cloud])
        
    return point_cloud
