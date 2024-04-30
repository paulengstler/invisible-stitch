import submitit
from submitit.helpers import Checkpointable, DelayedSubmission

import os
from enum import Enum
from typing import Optional

import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
import skimage
from scipy import ndimage

PLACES_PATH = "REPLACE_ME" # path to the places365 dataset
PLACES_DEPTH_PATH = "REPLACE_ME" # path to the computed places365 depth

class SlurmJobType(Enum):
    CPU = 0
    GPU = 1


class PointsRendererWithMasks(PointsRenderer):
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists
        weights = 1 - dists2 / (r * r)

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

def project_points(cameras, depth, use_pixel_centers=True):
    # only works with a single camera for now
    depth_t = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth
    depth_t = depth_t.to(cameras.device)

    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = cameras.focal_length[0, 1], cameras.focal_length[0, 0]
    cx, cy = cameras.principal_point[0, 1], cameras.principal_point[0, 0]

    # assume all cameras render images of the same size
    i, j = torch.meshgrid(
        torch.arange(cameras.image_size[0][0], dtype=torch.float32, device=cameras.device) + pixel_center,
        torch.arange(cameras.image_size[0][1], dtype=torch.float32, device=cameras.device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth_t / fx, -(j - cy) * depth_t / fy, depth_t], -1
    )

    directions_hom = torch.cat((directions.view(-1, 3), torch.ones_like(directions.view(-1, 3)[:, :1])), dim=1)
    xy_depth_world_hom = (directions_hom.to(cameras.device) @ cameras.get_world_to_view_transform().inverse().get_matrix())[0]
    xy_depth_world = xy_depth_world_hom[:, :3] / xy_depth_world_hom[:, 3:]
    xy_depth_world = (xy_depth_world.view(-1, 3)).unsqueeze(0)

    return xy_depth_world

def align_depths(previous_depth, new_depth, mask):
    assert previous_depth.ndim == mask.ndim == new_depth.ndim == 2
    valid_gt_depth = previous_depth[mask].view(-1).unsqueeze(1).clone()

    flat_new_depth = new_depth.to(mask.device)

    valid_pred_depth = flat_new_depth[mask].reshape(-1).unsqueeze(1)
    with torch.no_grad():
        A = torch.cat(
            [valid_pred_depth, torch.ones_like(valid_pred_depth)], dim=-1
        )  # [B, 2]
        X = torch.linalg.lstsq(A, valid_gt_depth).solution  # [2, 1]

        aligned_new_depth = flat_new_depth.reshape(-1).unsqueeze(1)

        aligned_new_depth = torch.cat(
            [aligned_new_depth, torch.ones_like(aligned_new_depth)], dim=-1
        ) @ X

    return aligned_new_depth

def align_depths_in_world(cameras, previous_depth, new_depth, mask):
    depth_coefficient = torch.tensor([1.0], requires_grad=True, device=device)

    depth_mask = (mask.view(-1) > 0) & (previous_depth.view(-1) > 0)

    original_points = project_points(cameras, previous_depth)[0]
    masked_original_points = original_points[depth_mask]

    optimizer = torch.optim.Adam((depth_coefficient,), lr=1e-2)
    previous_loss = torch.inf
    tolerance = 1e-5
    grace_period = 10
    for _ in range(10_000):
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(project_points(cameras, depth_coefficient * new_depth)[0][depth_mask], masked_original_points, reduction="mean")

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

    return (depth_coefficient * new_depth).detach()
    

def get_pointcloud(xy_depth_world, features=None, device="cpu"):
    point_cloud = Pointclouds(points=[xy_depth_world.to(device)], features=[features] if features is not None else None)
    return point_cloud

def merge_pointclouds(point_clouds):
    points = torch.cat([pc.points_padded() for pc in point_clouds], dim=1)
    features = torch.cat([pc.features_padded() for pc in point_clouds], dim=1)
    return Pointclouds(points=[points[0]], features=[features[0]])

def render(cameras, point_cloud):
    raster_settings = PointsRasterizationSettings(
        image_size=(int(cameras.image_size[0, 1]), int(cameras.image_size[0, 0])), 
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

def snap_high_gradients_to_nn(depth, threshold=20):
    grad_depth = np.copy(depth)
    grad_depth = grad_depth - grad_depth.min()
    grad_depth = grad_depth / grad_depth.max()

    grad = skimage.filters.rank.gradient(grad_depth, skimage.morphology.disk(1))
    return nearest_neighbor_fill(depth, torch.from_numpy(grad < threshold).to("cuda"))

def is_slurm_available() -> bool:
    return submitit.AutoExecutor(".").cluster == "slurm"


def setup_slurm(
    name: str,
    job_type: SlurmJobType,
    submitit_folder: str = "submitit",
    depend_on: Optional[str] = None,
    timeout: int = 180,
    high_compute_memory: bool = False,
) -> submitit.AutoExecutor:
    os.makedirs(submitit_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=submitit_folder, slurm_max_num_timeout=10)

    ################################################
    ##                                            ##
    ##   ADAPT THESE PARAMETERS TO YOUR CLUSTER   ##
    ##                                            ##
    ################################################

    # You may choose low-priority partitions where job preemption is enabled as
    # any preempted jobs will automatically resume/restart when rescheduled.

    if job_type == SlurmJobType.CPU:
        kwargs = {
            "slurm_partition": "compute",
            "gpus_per_node": 0,
            "slurm_cpus_per_task": 14,
            "slurm_mem": "32GB" if not high_compute_memory else "64GB",
        }
    elif job_type == SlurmJobType.GPU:
        kwargs = {
            "slurm_partition": "low-prio-gpu",
            "gpus_per_node": 1,
            "slurm_cpus_per_task": 4,
            "slurm_mem": "16GB",
            # If your cluster supports choosing specific GPUs based on constraints,
            # you can uncomment this line to select low-memory GPUs.
            "slurm_constraint": "p40",
        }

    ###################
    ##               ##
    ##   ALL DONE!   ##
    ##               ##
    ###################

    kwargs = {
        **kwargs,
        "slurm_job_name": name,
        "timeout_min": timeout,
        "tasks_per_node": 1,
        "slurm_additional_parameters": {"depend": f"afterany:{depend_on}"}
        if depend_on is not None
        else {},
    }

    executor.update_parameters(**kwargs)

    return executor

def run_inference_for_category(category_id, out_path):
    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm
    from torch.utils.data import Dataset, DataLoader

    class CategoryDataset(Dataset):
        def __init__(self, category_id, out_path):
            self.category_id = category_id

            self.category_path = os.path.join(PLACES_PATH, str(category_id))
            self.depth_path = os.path.join(PLACES_DEPTH_PATH, str(category_id))

            images_processed = len(os.listdir(os.path.join(out_path, str(category_id))))
            print(f"Found {images_processed} images that have already been processed")

            self.images = sorted(os.listdir(self.category_path))[images_processed:]
            self.depths = sorted(os.listdir(self.depth_path))[images_processed:]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image_name = self.images[idx]
            depth_name = self.depths[idx]

            image_path = os.path.join(self.category_path, image_name)
            depth_path = os.path.join(self.depth_path, depth_name)

            image = Image.open(image_path).convert("RGB")
            depth = np.load(depth_path)

            return image_name, image, depth

    print(f"This runner is for category {category_id}")

    os.makedirs(os.path.join(out_path, str(category_id)), exist_ok=True)

    dataset = CategoryDataset(category_id, out_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    device = "cuda"

    min_azim, max_azim = -15, 15
    min_elev, max_elev = -15, 15
    min_dist, max_dist = 0.9, 1.1

    for image_names in tqdm(dataloader):
        image_name, image, depth = image_names[0]

        snapped_depth = snap_high_gradients_to_nn(torch.from_numpy(depth), threshold=15).squeeze().cpu().numpy()

        image_size = image.size
        w, h = image_size
        focal_length = torch.tensor([w], dtype=torch.float32)

        oR, oT = look_at_view_transform(device=device)
        o_cameras = PerspectiveCameras(R=oR, T=oT, focal_length=focal_length, principal_point=(((h-1)/2, (w-1)/2),), image_size=(image_size,), device=device, in_ndc=False)

        rgb = (torch.from_numpy(np.asarray(image).copy()).reshape(-1, 3).float() / 255).to(device)

        point_cloud = get_pointcloud(project_points(o_cameras, snapped_depth)[0], features=rgb, device=device)

        randoms = torch.rand(3)
        azim = randoms[0] * (max_azim - min_azim) + min_azim
        elev = randoms[1] * (max_elev - min_elev) + min_elev
        dist = randoms[2] * (max_dist - min_dist) + min_dist

        R, T = look_at_view_transform(device=device, azim=azim, elev=elev, dist=dist)
        cameras = PerspectiveCameras(R=R, T=T, focal_length=focal_length, principal_point=(((h-1)/2, (w-1)/2),), image_size=(image_size,), device=device, in_ndc=False)

        with torch.no_grad():
            images, _, depths = render(cameras, point_cloud)

        depth_masks = (depths == torch.inf)

        depth_masks = depth_masks.squeeze().cpu().numpy().astype(np.uint8)

        Image.fromarray((images[0].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, str(category_id), image_name.replace(".jpg", ".png")))
        np.save(os.path.join(out_path, str(category_id), image_name.replace(".jpg", ".npy")), depth_masks)

class CategoryInference(Checkpointable):
    def __call__(self, *args, **kwargs):
        return run_inference_for_category(*args, **kwargs)

    def checkpoint(self, *args, **kwargs) -> DelayedSubmission:
        """Resubmits the same callable with the same arguments"""
        return DelayedSubmission(self, *args, **kwargs)  # type: ignore

def run_inference_for_all_categories(out_path):
    os.makedirs(out_path, exist_ok=True)

    category_ids = sorted(os.listdir(PLACES_PATH))

    if is_slurm_available():
        print("SLURM is available")

        executor = setup_slurm(
                    f"places365",
                    SlurmJobType.GPU,
                    timeout=24 * 60,
                )

        with executor.batch():
            for category_id in category_ids:
                executor.submit(CategoryInference(), category_id, out_path)

        print(f"Submitted {len(category_ids)} jobs to SLURM")

    else:
        from tqdm.auto import tqdm
        for category_id in tqdm(category_ids):
            run_inference_for_category(category_id, out_path)

    
def main(out_path):
    run_inference_for_all_categories(out_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
