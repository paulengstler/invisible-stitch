import numpy as np
import torch
import skimage
from scipy import ndimage
from PIL import Image
from .models import infer_with_zoe_dc
from pytorch3d.structures import Pointclouds

import math

def nearest_neighbor_fill(img, mask, erosion=0):
    img_ = np.copy(img.cpu().numpy())

    if erosion > 0:
        eroded_mask = skimage.morphology.binary_erosion(mask.cpu().numpy(), footprint=skimage.morphology.disk(erosion))
    else:
        eroded_mask = mask.cpu().numpy()

    img_[eroded_mask <= 0] = np.nan

    distance_to_boundary = ndimage.distance_transform_bf((~eroded_mask>0), metric="cityblock")

    for current_dist in np.unique(distance_to_boundary)[1:]:
        ii, jj = np.where(distance_to_boundary == current_dist)

        ii_ = np.array([ii - 1, ii, ii + 1, ii - 1, ii, ii + 1, ii - 1, ii, ii + 1]).reshape(9, -1)
        jj_ = np.array([jj - 1, jj - 1, jj - 1, jj, jj, jj, jj + 1, jj + 1, jj + 1]).reshape(9, -1)

        ii_ = ii_.clip(0, img_.shape[0] - 1)
        jj_ = jj_.clip(0, img_.shape[1] - 1)

        img_[ii, jj] = np.nanmax(img_[ii_, jj_], axis=0)

    return torch.from_numpy(img_).to(img.device)

def snap_high_gradients_to_nn(depth, threshold=20):
    grad_depth = np.copy(depth.cpu().numpy())
    grad_depth = grad_depth - grad_depth.min()
    grad_depth = grad_depth / grad_depth.max()

    grad = skimage.filters.rank.gradient(grad_depth, skimage.morphology.disk(1))
    return nearest_neighbor_fill(depth, torch.from_numpy(grad < threshold).to(depth.device), erosion=3)

def project_points(cameras, depth, use_pixel_centers=True):
    if len(cameras) > 1:
        import warnings
        warnings.warn("project_points assumes only a single camera is used")

    depth_t = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth
    depth_t = depth_t.to(cameras.device)

    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = cameras.focal_length[0, 1], cameras.focal_length[0, 0]
    cx, cy = cameras.principal_point[0, 1], cameras.principal_point[0, 0]

    i, j = torch.meshgrid(
        torch.arange(cameras.image_size[0][0], dtype=torch.float32, device=cameras.device) + pixel_center,
        torch.arange(cameras.image_size[0][1], dtype=torch.float32, device=cameras.device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth_t / fx, -(j - cy) * depth_t / fy, depth_t], -1
    )

    xy_depth_world = cameras.get_world_to_view_transform().inverse().transform_points(directions.view(-1, 3)).unsqueeze(0)

    return xy_depth_world

def get_pointcloud(xy_depth_world, device="cpu", features=None):
    point_cloud = Pointclouds(points=[xy_depth_world.to(device)], features=[features] if features is not None else None)
    return point_cloud

def merge_pointclouds(point_clouds):
    points = torch.cat([pc.points_padded() for pc in point_clouds], dim=1)
    features = torch.cat([pc.features_padded() for pc in point_clouds], dim=1)
    return Pointclouds(points=[points[0]], features=[features[0]])

def outpaint_with_depth_estimation(image, mask, previous_depth, h, w, pipe, zoe_dc, prompt, cameras, dilation_size: int = 2, depth_scaling: float = 1, generator = None):
    img_input = Image.fromarray((255*image[..., :3].cpu().numpy()).astype(np.uint8))

    # we slightly dilate the mask as aliasing might cause us to receive a too small mask from pytorch3d
    img_mask = Image.fromarray((255*skimage.morphology.isotropic_dilation(((~mask).cpu().numpy()), radius=dilation_size)).astype(np.uint8))#footprint=skimage.morphology.disk(dilation_size)))

    out_image = pipe(prompt=prompt, image=img_input, mask_image=img_mask, height=h, width=w, generator=generator).images[0]
    out_depth = infer_with_zoe_dc(zoe_dc, torch.from_numpy(np.asarray(out_image)/255.).permute(2,0,1).float().to(zoe_dc.device), (previous_depth * mask).to(zoe_dc.device), scaling=depth_scaling).cpu().numpy()

    return out_image, out_depth

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
