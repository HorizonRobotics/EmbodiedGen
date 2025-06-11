# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import base64
import logging
import math
import os
import sys
from glob import glob
from io import BytesIO
from typing import Union

import cv2
import imageio
import numpy as np
import PIL.Image as Image
import spaces
import torch
from moviepy.editor import VideoFileClip, clips_array
from tqdm import tqdm
from embodied_gen.data.differentiable_render import entrypoint as render_api

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, "../.."))
from thirdparty.TRELLIS.trellis.renderers.mesh_renderer import MeshRenderer
from thirdparty.TRELLIS.trellis.representations import MeshExtractResult
from thirdparty.TRELLIS.trellis.utils.render_utils import (
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "render_asset3d",
    "merge_images_video",
    "filter_small_connected_components",
    "filter_image_small_connected_components",
    "combine_images_to_base64",
    "render_mesh",
    "render_video",
]


@spaces.GPU
def render_asset3d(
    mesh_path: str,
    output_root: str,
    distance: float = 5.0,
    num_images: int = 1,
    elevation: list[float] = (0.0,),
    pbr_light_factor: float = 1.5,
    return_key: str = "image_color/*",
    output_subdir: str = "renders",
    gen_color_mp4: bool = False,
    gen_viewnormal_mp4: bool = False,
    gen_glonormal_mp4: bool = False,
) -> list[str]:
    input_args = dict(
        mesh_path=mesh_path,
        output_root=output_root,
        uuid=output_subdir,
        distance=distance,
        num_images=num_images,
        elevation=elevation,
        pbr_light_factor=pbr_light_factor,
        with_mtl=True,
    )
    if gen_color_mp4:
        input_args["gen_color_mp4"] = True
    if gen_viewnormal_mp4:
        input_args["gen_viewnormal_mp4"] = True
    if gen_glonormal_mp4:
        input_args["gen_glonormal_mp4"] = True
    try:
        _ = render_api(**input_args)
    except Exception as e:
        logger.error(f"Error occurred during rendering: {e}.")

    dst_paths = glob(os.path.join(output_root, output_subdir, return_key))

    return dst_paths


def merge_images_video(color_images, normal_images, output_path) -> None:
    width = color_images[0].shape[1]
    combined_video = [
        np.hstack([rgb_img[:, : width // 2], normal_img[:, width // 2 :]])
        for rgb_img, normal_img in zip(color_images, normal_images)
    ]
    imageio.mimsave(output_path, combined_video, fps=50)

    return


def merge_video_video(
    video_path1: str, video_path2: str, output_path: str
) -> None:
    """Merge two videos by the left half and the right half of the videos."""
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    if clip1.size != clip2.size:
        raise ValueError("The resolutions of the two videos do not match.")

    width, height = clip1.size
    clip1_half = clip1.crop(x1=0, y1=0, x2=width // 2, y2=height)
    clip2_half = clip2.crop(x1=width // 2, y1=0, x2=width, y2=height)
    final_clip = clips_array([[clip1_half, clip2_half]])
    final_clip.write_videofile(output_path, codec="libx264")


def filter_small_connected_components(
    mask: Union[Image.Image, np.ndarray],
    area_ratio: float,
    connectivity: int = 8,
) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=connectivity,
    )

    small_components = np.zeros_like(mask, dtype=np.uint8)
    mask_area = (mask != 0).sum()
    min_area = mask_area // area_ratio
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            small_components[labels == label] = 255

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(small_components))

    return mask


def filter_image_small_connected_components(
    image: Union[Image.Image, np.ndarray],
    area_ratio: float = 10,
    connectivity: int = 8,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = image.convert("RGBA")
        image = np.array(image)

    mask = image[..., 3]
    mask = filter_small_connected_components(mask, area_ratio, connectivity)
    image[..., 3] = mask

    return image


def combine_images_to_base64(
    images: list[str | Image.Image],
    cat_row_col: tuple[int, int] = None,
    target_wh: tuple[int, int] = (512, 512),
) -> str:
    n_images = len(images)
    if cat_row_col is None:
        n_col = math.ceil(math.sqrt(n_images))
        n_row = math.ceil(n_images / n_col)
    else:
        n_row, n_col = cat_row_col

    images = [
        Image.open(p).convert("RGB") if isinstance(p, str) else p
        for p in images[: n_row * n_col]
    ]
    images = [img.resize(target_wh) for img in images]

    grid_w, grid_h = n_col * target_wh[0], n_row * target_wh[1]
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, img in enumerate(images):
        row, col = divmod(idx, n_col)
        grid.paste(img, (col * target_wh[0], row * target_wh[1]))

    buffer = BytesIO()
    grid.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@spaces.GPU
def render_mesh(sample, extrinsics, intrinsics, options={}, **kwargs):
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = options.get("resolution", 512)
    renderer.rendering_options.near = options.get("near", 1)
    renderer.rendering_options.far = options.get("far", 100)
    renderer.rendering_options.ssaa = options.get("ssaa", 4)
    rets = {}
    for extr, intr in tqdm(zip(extrinsics, intrinsics), desc="Rendering"):
        res = renderer.render(sample, extr, intr)
        if "normal" not in rets:
            rets["normal"] = []
        normal = torch.lerp(
            torch.zeros_like(res["normal"]), res["normal"], res["mask"]
        )
        normal = np.clip(
            normal.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
        ).astype(np.uint8)
        rets["normal"].append(normal)

    return rets


@spaces.GPU
def render_video(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2,
    fov=40,
    **kwargs,
):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    yaws = yaws.tolist()
    pitch = [0.5] * num_frames
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitch, r, fov
    )
    render_fn = (
        render_mesh if isinstance(sample, MeshExtractResult) else render_frames
    )
    result = render_fn(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )

    return result


if __name__ == "__main__":
    # Example usage:
    merge_video_video(
        "outputs/imageto3d/room_bottle7/room_bottle_007/URDF_room_bottle_007/mesh_glo_normal.mp4",  # noqa
        "outputs/imageto3d/room_bottle7/room_bottle_007/URDF_room_bottle_007/mesh.mp4",  # noqa
        "merge.mp4",
    )

    image_base64 = combine_images_to_base64(
        [
            "apps/assets/example_image/sample_00.jpg",
            "apps/assets/example_image/sample_01.jpg",
            "apps/assets/example_image/sample_02.jpg",
        ]
    )
