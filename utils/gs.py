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

from tqdm.auto import tqdm

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

    # NOTE: This is a temporary "fix", sorry. With this loop, the resulting Gaussian splatting scenes
    #       appear very fuzzy. Once time permits, I will update this loop with a nerfstudio/gsplat
    #       implementation that optimizes the cameras, leading to a notably sharper result.

    return scene

    from random import randint
    from .gaussian_renderer import render as gs_render
    from .scene.utils.loss_utils import l1_loss, ssim

    pbar = tqdm(range(1, gs_options.iterations + 1))
    for iteration in pbar:
        scene.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            scene.gaussians.oneupSHdegree()

        # Pick a random Camera
        random_idx = randint(0, len(gs_optimization_bundle["frames"])-1)
        viewpoint_cam = scene.getTrainCameras()[random_idx]

        # Render
        render_pkg = gs_render(viewpoint_cam, scene.gaussians, gs_options, scene.background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image, reduce=False)
        loss = (1.0 - gs_options.lambda_dssim) * Ll1

        if viewpoint_cam.mask is not None:
            mask = torch.from_numpy(viewpoint_cam.mask).to(loss.device)
        else:
            mask = 1

        loss = (loss * mask).mean()
        loss = loss + gs_options.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        pbar.set_description(f"Loss: {loss.item():.4f}")

        with torch.no_grad():
            # Densification
            if iteration < gs_options.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                scene.gaussians.max_radii2D[visibility_filter] = torch.max(
                    scene.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                scene.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > gs_options.densify_from_iter and iteration % gs_options.densification_interval == 0:
                    size_threshold = 20 if iteration > gs_options.opacity_reset_interval else None
                    scene.gaussians.densify_and_prune(
                        gs_options.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if (iteration % gs_options.opacity_reset_interval == 0 
                    or (gs_options.white_background and iteration == gs_options.densify_from_iter)
                ):
                    scene.gaussians.reset_opacity()

            # Optimizer step
            if iteration < gs_options.iterations:
                scene.gaussians.optimizer.step()
                scene.gaussians.optimizer.zero_grad(set_to_none = True)

    return scene

gs_options = edict({
    "sh_degree": 3,
    "images": "images",
    "resolution": -1,
    "white_background": False,
    "data_device": "cuda",
    "eval": False,
    "use_depth": False,
    "iterations": 0,#250,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 2990,
    "feature_lr": 0.0,#0.0025,
    "opacity_lr": 0.0,#0.05,
    "scaling_lr": 0.0,#0.005,
    "rotation_lr": 0.0,#0.001,
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
