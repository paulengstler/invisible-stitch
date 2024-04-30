import glob
import os

import torch
import torch.nn.functional as F
import numpy as np

from zoedepth.utils.misc import colorize
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.models.model_io import load_wts

from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline

def load_ckpt(config, model, checkpoint_dir: str = "./checkpoints", ckpt_type: str = "best"):
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

def get_zoe_dc_model(vanilla: bool = False, ckpt_path: str = None, **kwargs):
    def ZoeD_N(midas_model_type="DPT_BEiT_L_384", vanilla=False, **kwargs):
        if midas_model_type != "DPT_BEiT_L_384":
            raise ValueError(f"Only DPT_BEiT_L_384 MiDaS model is supported for pretrained Zoe_N model, got: {midas_model_type}")

        zoedepth_config = get_config("zoedepth", "train", **kwargs)
        model = build_model(zoedepth_config)

        if vanilla:
            model.__setattr__("vanilla", True)
            return model
        else:
            model.__setattr__("vanilla", False)

        if zoedepth_config.add_depth_channel and not vanilla:
            model.core.core.pretrained.model.patch_embed.proj = torch.nn.Conv2d(
                model.core.core.pretrained.model.patch_embed.proj.in_channels+2,
                model.core.core.pretrained.model.patch_embed.proj.out_channels,
                kernel_size=model.core.core.pretrained.model.patch_embed.proj.kernel_size,
                stride=model.core.core.pretrained.model.patch_embed.proj.stride,
                padding=model.core.core.pretrained.model.patch_embed.proj.padding,
                bias=True)

        if ckpt_path is not None:
            assert os.path.exists(ckpt_path)
            zoedepth_config.__setattr__("checkpoint", ckpt_path)
        else:
            assert vanilla, "ckpt_path must be provided for non-vanilla model"

        model = load_ckpt(zoedepth_config, model)

        return model

    return ZoeD_N(vanilla=vanilla, ckpt_path=ckpt_path, **kwargs)

def infer_with_pad(zoe, x, pad_input: bool = True, fh: float = 3, fw: float = 3, upsampling_mode: str = "bicubic", padding_mode: str = "reflect", **kwargs):
    assert x.dim() == 4, "x must be 4 dimensional, got {}".format(x.dim())

    if pad_input:
        assert fh > 0 or fw > 0, "atlease one of fh and fw must be greater than 0"
        pad_h = int(np.sqrt(x.shape[2]/2) * fh)
        pad_w = int(np.sqrt(x.shape[3]/2) * fw)
        padding = [pad_w, pad_w]
        if pad_h > 0:
            padding += [pad_h, pad_h]
        
        x_rgb = x[:, :3]
        x_remaining = x[:, 3:]
        x_rgb = F.pad(x_rgb, padding, mode=padding_mode, **kwargs)
        x_remaining = F.pad(x_remaining, padding, mode="constant", value=0, **kwargs)
        x = torch.cat([x_rgb, x_remaining], dim=1)
    out = zoe(x)["metric_depth"]
    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode=upsampling_mode, align_corners=False)
    if pad_input:
        # crop to the original size, handling the case where pad_h and pad_w is 0
        if pad_h > 0:
            out = out[:, :, pad_h:-pad_h,:]
        if pad_w > 0:
            out = out[:, :, :, pad_w:-pad_w]
    return out

@torch.no_grad()
def infer_with_zoe_dc(zoe_dc, image, sparse_depth, scaling: float = 1):
    sparse_depth_mask = (sparse_depth[None, None, ...] > 0).float()
    # the metric depth range defined during training is [1e-3, 10]
    x = torch.cat([image[None, ...], sparse_depth[None, None, ...] / (float(scaling) * 10.0), sparse_depth_mask], dim=1).to(zoe_dc.device)

    out = infer_with_pad(zoe_dc, x)
    out_flip = infer_with_pad(zoe_dc, torch.flip(x, dims=[3]))
    out = (out + torch.flip(out_flip, dims=[3])) / 2

    pred_depth = float(scaling) * out

    return torch.nn.functional.interpolate(pred_depth, image.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

def get_sd_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.vae = AsymmetricAutoencoderKL.from_pretrained(
        "cross-attention/asymmetric-autoencoder-kl-x-2", 
        torch_dtype=torch.float16
    )

    return pipe
