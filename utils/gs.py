import random
import torch
import numpy as np
from .scene import GaussianModel
from .scene.dataset_readers import SceneInfo, getNerfppNorm
from .scene.cameras import Camera
from .ops import focal2fov, fov2focal
from .scene.gaussian_model import BasicPointCloud
from easydict import EasyDict as edict
from PIL import Image

from gsplat.sh import num_sh_bases, spherical_harmonics
from gsplat import project_gaussians, rasterize_gaussians

from tqdm.auto import tqdm
from torch import Tensor

# All credit for this function goes to the nerfstudio project
# nerfstudio/cameras/lie_groups.py
def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 4, 3, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, 3, :3] = tangent_vector[:, :3]
    return ret

def get_blank_gs_bundle(h, w):
    return {
        "camera_angle_x": focal2fov(torch.tensor([w], dtype=torch.float32), w),
        "W": w,
        "H": h,
        "pcd_points": None,
        "pcd_colors": None,
        'frames': [],
    }

def read_cameras_from_optimization_bundle(optimization_bundle, white_background: bool = False):
    cameras = []

    fovx = optimization_bundle["camera_angle_x"]
    frames = optimization_bundle["frames"]

    # we flip the x and y axis to move from PyTorch3D's coordinate system to COLMAP's
    coordinate_system_transform = np.array([-1, -1, 1])

    for idx, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"])
        c2w[:3, :3] = c2w[:3, :3] * coordinate_system_transform

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = c2w[-1, :3] * coordinate_system_transform

        image = frame["image"]

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        image = torch.Tensor(arr).permute(2,0,1)

        cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, mask=frame.get("mask", None),
                                gt_alpha_mask=None, image_name='', uid=idx, data_device='cuda'))

    return cameras

class Scene:
    gaussians: GaussianModel

    def __init__(self, traindata, gaussians: GaussianModel, gs_options, shuffle: bool = True):
        self.traindata = traindata
        self.gaussians = gaussians
        
        train_cameras = read_cameras_from_optimization_bundle(traindata, gs_options.white_background)
        
        nerf_normalization = getNerfppNorm(train_cameras)

        pcd = BasicPointCloud(points=traindata['pcd_points'], colors=traindata['pcd_colors'], normals=None)
        
        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=train_cameras,
                               test_cameras=[],
                               nerf_normalization=nerf_normalization,
                               ply_path='')

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.train_cameras = scene_info.train_cameras

        bg_color = np.array([1,1,1]) if gs_options.white_background else np.array([0, 0, 0])
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        self.gaussians.training_setup(gs_options)

    def getTrainCameras(self):
        return self.train_cameras
    
    def getPresetCameras(self, preset):
        assert preset in self.preset_cameras
        return self.preset_cameras[preset]

def run_gaussian_splatting(scene, gs_optimization_bundle):
    torch.cuda.empty_cache()

    scene.gaussians._opacity = torch.ones_like(scene.gaussians._opacity)

    from random import randint
    from .scene.utils.loss_utils import l1_loss, ssim

    pose_adjustments = torch.nn.Parameter(torch.zeros((len(scene.getTrainCameras()), 6), device=scene.getTrainCameras()[0].world_view_transform.device), requires_grad=True)
    pose_optimizer = torch.optim.Adam([pose_adjustments], lr=1e-4)

    pbar = tqdm(range(1, gs_options.iterations + 1))
    for iteration in pbar:
        scene.gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        random_idx = randint(0, len(gs_optimization_bundle["frames"])-1)
        viewpoint_cam = scene.getTrainCameras()[random_idx] # FIXME

        adj = exp_map_SO3xR3(pose_adjustments[random_idx, :][None, :])
        adj = torch.cat([adj, torch.Tensor([0, 0, 0, 1])[None, :, None].to(adj)], dim=2)
        w2v = torch.bmm(viewpoint_cam.world_view_transform[None, ...], adj)[0]

        R = w2v[:3, :3]  # 3 x 3
        T = w2v[3, :3].view(3, 1)  # 3 x 1

        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R
        viewmat[:3, 3:4] = T
        H, W = viewpoint_cam.original_image.shape[1:]

        cx = W / 2
        cy = H / 2
        fx = fov2focal(viewpoint_cam.FoVx, W)
        fy = fov2focal(viewpoint_cam.FoVy, H)

        colors = torch.cat((scene.gaussians._features_dc, scene.gaussians._features_rest), dim=1)
        BLOCK_WIDTH = 16
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            scene.gaussians._xyz,
            torch.exp(scene.gaussians._scaling),
            1,
            scene.gaussians._rotation / scene.gaussians._rotation.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            fx,
            fy,
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )

        #xys.retain_grad()

        viewdirs = scene.gaussians._xyz.detach() - viewpoint_cam.world_view_transform[3, :3]  # (N, 3)
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(gs_options.sh_degree, viewdirs, colors)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        opacities = torch.sigmoid(scene.gaussians._opacity) * comp[:, None]

        rgb, alpha = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=torch.zeros(3, device=R.device),
            return_alpha=True,
        )  # type: ignore
        rgb = torch.clamp(rgb, max=1.0)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(rgb.permute(2, 0, 1), gt_image, reduce=False)
        loss = (1.0 - gs_options.lambda_dssim) * Ll1

        if viewpoint_cam.mask is not None:
            mask = torch.from_numpy(viewpoint_cam.mask).to(loss.device)
        else:
            mask = 1

        loss = (loss * mask).mean()
        loss = loss + gs_options.lambda_dssim * (1.0 - ssim(rgb.permute(2, 0, 1), gt_image))
        loss.backward()

        pbar.set_description(f"Loss: {loss.item():.4f}")

        # Optimizer step
        scene.gaussians.optimizer.step()
        scene.gaussians.optimizer.zero_grad(set_to_none=True)

        pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)

    return scene

gs_options = edict({
    "sh_degree": 3,
    "images": "images",
    "resolution": -1,
    "white_background": False,
    "data_device": "cuda",
    "eval": False,
    "use_depth": False,
    "iterations": 990,
    "position_lr_init": 1.6e-12,
    "position_lr_final": 1.6e-12,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 2990,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "percent_dense": 0.01,
    "lambda_dssim": 0.2,
    "densification_interval": 100,
    "opacity_reset_interval": 3000,
    "densify_from_iter": 10_000,
    "densify_until_iter": 15_000,
    "densify_grad_threshold": 0.0002,
    "convert_SHs_python": False,
    "compute_cov3D_python": False,
    "debug": False,
})
