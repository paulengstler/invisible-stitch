import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import skimage
from PIL import Image

import gradio as gr

from utils.render import PointsRendererWithMasks, render
from utils.ops import snap_high_gradients_to_nn, project_points, get_pointcloud, merge_pointclouds, outpaint_with_depth_estimation
from utils.gs import gs_options, read_cameras_from_optimization_bundle, Scene, run_gaussian_splatting, get_blank_gs_bundle

from pytorch3d.utils import opencv_from_cameras_projection
from utils.ops import focal2fov, fov2focal
from utils.models import infer_with_zoe_dc
from utils.scene import GaussianModel
from utils.demo import downsample_point_cloud
from typing import Iterable, Tuple, Dict, Optional
import itertools

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
)

from pytorch3d.io import IO

def get_blank_gs_bundle(h, w):
    return {
        "camera_angle_x": focal2fov(torch.tensor([w], dtype=torch.float32), w),
        "W": w,
        "H": h,
        "pcd_points": None,
        "pcd_colors": None,
        'frames': [],
    }

def extrapolate_point_cloud(prompt: str, image_size: Tuple[int, int], look_at_params: Iterable[Tuple[float, float, float, Tuple[float, float, float]]], point_cloud: Pointclouds = None, dry_run: bool = False, discard_mask: bool = False, initial_image: Optional[Image.Image] = None, depth_scaling: float = 1, **render_kwargs):
    w, h = image_size
    optimization_bundle_frames = []

    for azim, elev, dist, at in look_at_params:
        R, T = look_at_view_transform(device=device, azim=azim, elev=elev, dist=dist, at=at)
        cameras = PerspectiveCameras(R=R, T=T, focal_length=torch.tensor([w], dtype=torch.float32), principal_point=(((h-1)/2, (w-1)/2),), image_size=(image_size,), device=device, in_ndc=False)

        if point_cloud is not None:
            images, masks, depths = render(cameras, point_cloud, **render_kwargs)

            if not dry_run:
                eroded_mask = skimage.morphology.binary_erosion((depths[0] > 0).cpu().numpy(), footprint=None)#skimage.morphology.disk(1))
                eroded_depth = depths[0].clone()
                eroded_depth[torch.from_numpy(eroded_mask).to(depths.device) <= 0] = 0

                outpainted_img, aligned_depth = outpaint_with_depth_estimation(images[0], masks[0], eroded_depth, h, w, pipe, zoe_dc_model, prompt, cameras, dilation_size=2, depth_scaling=depth_scaling, generator=torch.Generator(device=pipe.device).manual_seed(0))

                aligned_depth = torch.from_numpy(aligned_depth).to(device)

            else:
                # in a dry run, we do not actually outpaint the image
                outpainted_img = Image.fromarray((255*images[0].cpu().numpy()).astype(np.uint8))

        else:
            assert initial_image is not None
            assert not dry_run

            # jumpstart the point cloud with a regular depth estimation
            t_initial_image = torch.from_numpy(np.asarray(initial_image)/255.).permute(2,0,1).float()
            depth = aligned_depth = infer_with_zoe_dc(zoe_dc_model, t_initial_image, torch.zeros(h, w))
            outpainted_img = initial_image
            images = [t_initial_image.to(device)]
            masks = [torch.ones(h, w, dtype=torch.bool).to(device)]

        if not dry_run:
            # snap high gradients to nearest neighbor, which eliminates noodle artifacts
            aligned_depth = snap_high_gradients_to_nn(aligned_depth, threshold=12).cpu()
            xy_depth_world = project_points(cameras, aligned_depth)

        c2w = cameras.get_world_to_view_transform().get_matrix()[0]

        optimization_bundle_frames.append({
            "image": outpainted_img,
            "mask": masks[0].cpu().numpy(),
            "transform_matrix": c2w.tolist(),
            "azim": azim,
            "elev": elev,
            "dist": dist,
        })

        if discard_mask:
            optimization_bundle_frames[-1].pop("mask")

        if not dry_run:
            optimization_bundle_frames[-1]["center_point"] = xy_depth_world[0].mean(dim=0).tolist()
            optimization_bundle_frames[-1]["depth"] = aligned_depth.cpu().numpy()
            optimization_bundle_frames[-1]["mean_depth"] = aligned_depth.mean().item()

        else:
            # in a dry run, we do not modify the point cloud
            continue

        rgb = (torch.from_numpy(np.asarray(outpainted_img).copy()).reshape(-1, 3).float() / 255).to(device)

        if point_cloud is None:
            point_cloud = get_pointcloud(xy_depth_world[0], device=device, features=rgb)

        else:
            # pytorch 3d's mask might be slightly too big (subpixels), so we erode it a little to avoid seams
            # in theory, 1 pixel is sufficient but we use 2 to be safe
            masks[0] = torch.from_numpy(skimage.morphology.binary_erosion(masks[0].cpu().numpy(), footprint=skimage.morphology.disk(2))).to(device)

            partial_outpainted_point_cloud = get_pointcloud(xy_depth_world[0][~masks[0].view(-1)], device=device, features=rgb[~masks[0].view(-1)])

            point_cloud = merge_pointclouds([point_cloud, partial_outpainted_point_cloud])

    return optimization_bundle_frames, point_cloud

def generate_point_cloud(initial_image: Image.Image, prompt: str):
    image_size = initial_image.size
    w, h = image_size

    optimization_bundle = get_blank_gs_bundle(h, w)

    step_size = 25

    azim_steps = [0, step_size, -step_size]
    look_at_params = [(azim, 0, 0.01, torch.zeros((1, 3))) for azim in azim_steps]

    optimization_bundle["frames"], point_cloud = extrapolate_point_cloud(prompt, image_size, look_at_params, discard_mask=True, initial_image=initial_image, depth_scaling=0.5, fill_point_cloud_holes=True)

    optimization_bundle["pcd_points"] = point_cloud.points_padded()[0].cpu().numpy()
    optimization_bundle["pcd_colors"] = point_cloud.features_padded()[0].cpu().numpy()

    return optimization_bundle, point_cloud

def supplement_point_cloud(optimization_bundle: Dict, point_cloud: Pointclouds, prompt: str):
    w, h = optimization_bundle["W"], optimization_bundle["H"]

    supporting_frames = []

    for i, frame in enumerate(optimization_bundle["frames"]):
        # skip supporting views
        if frame.get("supporting", False):
            continue

        center_point = torch.tensor(frame["center_point"]).to(device)
        mean_depth = frame["mean_depth"]
        azim, elev = frame["azim"], frame["elev"]

        azim_jitters = torch.linspace(-5, 5, 3).tolist()
        elev_jitters = torch.linspace(-5, 5, 3).tolist()

        # build the product of azim and elev jitters
        camera_jitters = [{"azim": azim + azim_jitter, "elev": elev + elev_jitter} for azim_jitter, elev_jitter in itertools.product(azim_jitters, elev_jitters)]

        look_at_params = [(camera_jitter["azim"], camera_jitter["elev"], mean_depth, center_point.unsqueeze(0)) for camera_jitter in camera_jitters]

        local_supporting_frames, point_cloud = extrapolate_point_cloud(prompt, (w, h), look_at_params, point_cloud, dry_run=True, depth_scaling=0.5, antialiasing=3)

        for local_supporting_frame in local_supporting_frames:
            local_supporting_frame["supporting"] = True

        supporting_frames.extend(local_supporting_frames)

    optimization_bundle["pcd_points"] = point_cloud.points_padded()[0].cpu().numpy()
    optimization_bundle["pcd_colors"] = point_cloud.features_padded()[0].cpu().numpy()

    return optimization_bundle, point_cloud

def generate_scene(img: Image.Image, prompt: str):
    assert isinstance(img, Image.Image)

    # resize image maintaining the aspect ratio so the longest side is 720 pixels
    max_size = 720
    img.thumbnail((max_size, max_size))

    # crop to ensure the image dimensions are divisible by 8
    img = img.crop((0, 0, img.width - img.width % 8, img.height - img.height % 8))

    from hashlib import sha1
    from datetime import datetime

    run_id = sha1(datetime.now().isoformat().encode()).hexdigest()[:6]

    run_name = f"gradio_{run_id}"

    gs_optimization_bundle, point_cloud = generate_point_cloud(img, prompt)

    #downsampled_point_cloud = downsample_point_cloud(gs_optimization_bundle, device=device)

    #gs_optimization_bundle["pcd_points"] = downsampled_point_cloud.points_padded()[0].cpu().numpy()
    #gs_optimization_bundle["pcd_colors"] = downsampled_point_cloud.features_padded()[0].cpu().numpy()

    scene = Scene(gs_optimization_bundle, GaussianModel(gs_options.sh_degree), gs_options)

    scene = run_gaussian_splatting(scene, gs_optimization_bundle)

    # coordinate system transformation
    scene.gaussians._xyz = scene.gaussians._xyz.detach()
    scene.gaussians._xyz[:, 1] = -scene.gaussians._xyz[:, 1]
    scene.gaussians._xyz[:, 2] = -scene.gaussians._xyz[:, 2]

    os.makedirs("outputs", exist_ok=True)
    save_path = os.path.join("outputs", f"{run_name}.ply")

    scene.gaussians.save_ply(save_path)

    return save_path

if __name__ == "__main__":
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    from utils.models import get_zoe_dc_model, get_sd_pipeline

    global zoe_dc_model
    from huggingface_hub import hf_hub_download
    zoe_dc_model = get_zoe_dc_model(ckpt_path=hf_hub_download(repo_id="paulengstler/invisible-stitch", filename="invisible-stitch.pt")).to(device)

    global pipe
    pipe = get_sd_pipeline(device)

    demo = gr.Interface(
        fn=generate_scene,
        inputs=[
            gr.Image(label="Input Image", sources=["upload", "clipboard"], type="pil"),
            gr.Textbox(label="Scene Hallucination Prompt")
        ],
        outputs=gr.Model3D(label="Generated Scene"),
        allow_flagging="never",
        title="Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting",
        description="Hallucinate geometrically coherent 3D scenes from a single input image in less than 30 seconds.<br /> [Project Page](https://research.paulengstler.com/invisible-stitch) | [GitHub](https://github.com/paulengstler/invisible-stitch) | [Paper](#) <br /><br />To keep this demo snappy, we have limited its functionality. Scenes are generated at a low resolution without densification, supporting views are not inpainted, and we do not optimize the resulting point cloud. Imperfections are to be expected, in particular around object borders. Please allow a couple of seconds for the generated scene to be downloaded (about 40 megabytes).",
        article="Please consider running this demo locally to obtain high-quality results (see the GitHub repository).<br /><br />Here are some observations we made that might help you to get better results:<ul><li>Use generic prompts that match the surroundings of your input image.</li><li>Ensure that the borders of your input image are free from partially visible objects.</li><li>Keep your prompts simple and avoid adding specific details.</li></ul>",
        examples=[
            ["examples/photo-1667788000333-4e36f948de9a.jpeg", "a street with traditional buildings in Kyoto, Japan"],
            ["examples/photo-1628624747186-a941c476b7ef.jpeg", "a suburban street in North Carolina on a bright, sunny day"],
            ["examples/photo-1469559845082-95b66baaf023.jpeg", "a view of Zion National Park"],
            ["examples/photo-1514984879728-be0aff75a6e8.jpeg", "a close-up view of a muddy path in a forest"],
            ["examples/photo-1618197345638-d2df92b39fe1.jpeg", "a close-up view of a white linen bed in a minimalistic room"],
            ["examples/photo-1546975490-e8b92a360b24.jpeg", "a warm living room with plants"],
            ["examples/photo-1499916078039-922301b0eb9b.jpeg", "a cozy bedroom on a bright day"],
        ])
    demo.queue().launch(share=True)
