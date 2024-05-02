import torch
import skimage
from pytorch3d.structures import Pointclouds
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
from .ops import nearest_neighbor_fill

from typing import cast, Optional

class PointsRendererWithMasks(PointsRenderer):
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists
        weights = torch.ones_like(dists2)#1 - dists2 / (r * r)
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

def render_with_settings(cameras, point_cloud, raster_settings, antialiasing: int = 1):
    if antialiasing > 1:
        raster_settings.image_size = (raster_settings.image_size[0] * antialiasing, raster_settings.image_size[1] * antialiasing)

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer = PointsRendererWithMasks(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    if antialiasing > 1:
        images, masks, depths = renderer(point_cloud)

        images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        images = F.avg_pool2d(images, kernel_size=antialiasing, stride=antialiasing)
        images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC

    else:
        return renderer(point_cloud)


def render(cameras, point_cloud, fill_point_cloud_holes: bool = False, radius: Optional[float] = None, antialiasing: int = 1):
    if fill_point_cloud_holes:
        coarse_raster_settings = PointsRasterizationSettings(
            image_size=(int(cameras.image_size[0, 1]), int(cameras.image_size[0, 0])), 
            radius = 1e-2,
            points_per_pixel = 1
        )

        _, coarse_mask, _ = render_with_settings(cameras, point_cloud, coarse_raster_settings)

        eroded_coarse_mask = torch.from_numpy(skimage.morphology.binary_erosion(coarse_mask[0].cpu().numpy(), footprint=skimage.morphology.disk(2)))

        raster_settings = PointsRasterizationSettings(
            image_size=(int(cameras.image_size[0, 1]), int(cameras.image_size[0, 0])), 
            radius = (1 / float(max(cameras.image_size[0, 1], cameras.image_size[0, 0])) * 2.0) if radius is None else radius,
            points_per_pixel = 16
        )

        # Render the scene
        images, masks, depths = render_with_settings(cameras, point_cloud, raster_settings)

        holes_in_rendering = masks[0].cpu() ^ eroded_coarse_mask

        images[0] = nearest_neighbor_fill(images[0], ~holes_in_rendering, 0)
        depths[0] = nearest_neighbor_fill(depths[0], ~holes_in_rendering, 0)

        return images, eroded_coarse_mask.unsqueeze(0).to(masks.device), depths

    else:
        raster_settings = PointsRasterizationSettings(
            image_size=(int(cameras.image_size[0, 1]), int(cameras.image_size[0, 0])), 
            radius = (1 / float(max(cameras.image_size[0, 1], cameras.image_size[0, 0])) * 2.0) if radius is None else radius,
            points_per_pixel = 16
        )

        # Render the scene
        return render_with_settings(cameras, point_cloud, raster_settings)
