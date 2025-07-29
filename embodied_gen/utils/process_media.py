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


import logging
import math
import mimetypes
import os
import textwrap
from glob import glob
from typing import Union

import cv2
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import spaces
from matplotlib.patches import Patch
from moviepy.editor import VideoFileClip, clips_array
from PIL import Image
from embodied_gen.data.differentiable_render import entrypoint as render_api
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "render_asset3d",
    "merge_images_video",
    "filter_small_connected_components",
    "filter_image_small_connected_components",
    "combine_images_to_grid",
    "SceneTreeVisualizer",
    "is_image_file",
    "parse_text_prompts",
    "check_object_edge_truncated",
]


@spaces.GPU
def render_asset3d(
    mesh_path: str,
    output_root: str,
    distance: float = 5.0,
    num_images: int = 1,
    elevation: list[float] = (0.0,),
    pbr_light_factor: float = 1.2,
    return_key: str = "image_color/*",
    output_subdir: str = "renders",
    gen_color_mp4: bool = False,
    gen_viewnormal_mp4: bool = False,
    gen_glonormal_mp4: bool = False,
    no_index_file: bool = False,
    with_mtl: bool = True,
) -> list[str]:
    input_args = dict(
        mesh_path=mesh_path,
        output_root=output_root,
        uuid=output_subdir,
        distance=distance,
        num_images=num_images,
        elevation=elevation,
        pbr_light_factor=pbr_light_factor,
        with_mtl=with_mtl,
        gen_color_mp4=gen_color_mp4,
        gen_viewnormal_mp4=gen_viewnormal_mp4,
        gen_glonormal_mp4=gen_glonormal_mp4,
        no_index_file=no_index_file,
    )

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


def combine_images_to_grid(
    images: list[str | Image.Image],
    cat_row_col: tuple[int, int] = None,
    target_wh: tuple[int, int] = (512, 512),
) -> list[Image.Image]:
    n_images = len(images)
    if n_images == 1:
        return images

    if cat_row_col is None:
        n_col = math.ceil(math.sqrt(n_images))
        n_row = math.ceil(n_images / n_col)
    else:
        n_row, n_col = cat_row_col

    images = [
        Image.open(p).convert("RGB") if isinstance(p, str) else p
        for p in images
    ]
    images = [img.resize(target_wh) for img in images]

    grid_w, grid_h = n_col * target_wh[0], n_row * target_wh[1]
    grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

    for idx, img in enumerate(images):
        row, col = divmod(idx, n_col)
        grid.paste(img, (col * target_wh[0], row * target_wh[1]))

    return [grid]


class SceneTreeVisualizer:
    def __init__(self, layout_info: LayoutInfo) -> None:
        self.tree = layout_info.tree
        self.relation = layout_info.relation
        self.objs_desc = layout_info.objs_desc
        self.G = nx.DiGraph()
        self.root = self._find_root()
        self._build_graph()

        self.role_colors = {
            Scene3DItemEnum.BACKGROUND.value: "plum",
            Scene3DItemEnum.CONTEXT.value: "lightblue",
            Scene3DItemEnum.ROBOT.value: "lightcoral",
            Scene3DItemEnum.MANIPULATED_OBJS.value: "lightgreen",
            Scene3DItemEnum.DISTRACTOR_OBJS.value: "lightgray",
            Scene3DItemEnum.OTHERS.value: "orange",
        }

    def _find_root(self) -> str:
        children = {c for cs in self.tree.values() for c, _ in cs}
        parents = set(self.tree.keys())
        roots = parents - children
        if not roots:
            raise ValueError("No root node found.")
        return next(iter(roots))

    def _build_graph(self):
        for parent, children in self.tree.items():
            for child, relation in children:
                self.G.add_edge(parent, child, relation=relation)

    def _get_node_role(self, node: str) -> str:
        if node == self.relation.get(Scene3DItemEnum.BACKGROUND.value):
            return Scene3DItemEnum.BACKGROUND.value
        if node == self.relation.get(Scene3DItemEnum.CONTEXT.value):
            return Scene3DItemEnum.CONTEXT.value
        if node == self.relation.get(Scene3DItemEnum.ROBOT.value):
            return Scene3DItemEnum.ROBOT.value
        if node in self.relation.get(
            Scene3DItemEnum.MANIPULATED_OBJS.value, []
        ):
            return Scene3DItemEnum.MANIPULATED_OBJS.value
        if node in self.relation.get(
            Scene3DItemEnum.DISTRACTOR_OBJS.value, []
        ):
            return Scene3DItemEnum.DISTRACTOR_OBJS.value
        return Scene3DItemEnum.OTHERS.value

    def _get_positions(
        self, root, width=1.0, vert_gap=0.1, vert_loc=1, xcenter=0.5, pos=None
    ):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        children = list(self.G.successors(root))
        if children:
            dx = width / len(children)
            next_x = xcenter - width / 2 - dx / 2
            for child in children:
                next_x += dx
                pos = self._get_positions(
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=next_x,
                    pos=pos,
                )
        return pos

    def render(
        self,
        save_path: str,
        figsize=(8, 6),
        dpi=300,
        title: str = "Scene 3D Hierarchy Tree",
    ):
        node_colors = [
            self.role_colors[self._get_node_role(n)] for n in self.G.nodes
        ]
        pos = self._get_positions(self.root)

        plt.figure(figsize=figsize)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            arrows=False,
            node_size=2000,
            node_color=node_colors,
            font_size=10,
            font_weight="bold",
        )

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.G, "relation")
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels=edge_labels,
            font_size=9,
            font_color="black",
        )

        # Draw small description text under each node (if available)
        for node, (x, y) in pos.items():
            desc = self.objs_desc.get(node)
            if desc:
                wrapped = "\n".join(textwrap.wrap(desc, width=30))
                plt.text(
                    x,
                    y - 0.006,
                    wrapped,
                    fontsize=6,
                    ha="center",
                    va="top",
                    wrap=True,
                    color="black",
                    bbox=dict(
                        facecolor="dimgray",
                        edgecolor="darkgray",
                        alpha=0.1,
                        boxstyle="round,pad=0.2",
                    ),
                )

        plt.title(title, fontsize=12)
        task_desc = self.relation.get("task_desc", "")
        if task_desc:
            plt.suptitle(
                f"Task Description: {task_desc}", fontsize=10, y=0.999
            )

        plt.axis("off")

        legend_handles = [
            Patch(facecolor=color, edgecolor='black', label=role)
            for role, color in self.role_colors.items()
        ]
        plt.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.1),
            fontsize=9,
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()


def load_scene_dict(file_path: str) -> dict:
    scene_dict = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            scene_id, desc = line.split(":", 1)
            scene_dict[scene_id.strip()] = desc.strip()

    return scene_dict


def is_image_file(filename: str) -> bool:
    mime_type, _ = mimetypes.guess_type(filename)

    return mime_type is not None and mime_type.startswith('image')


def parse_text_prompts(prompts: list[str]) -> list[str]:
    if len(prompts) == 1 and prompts[0].endswith(".txt"):
        with open(prompts[0], "r") as f:
            prompts = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    return prompts


def alpha_blend_rgba(
    fg_image: Union[str, Image.Image, np.ndarray],
    bg_image: Union[str, Image.Image, np.ndarray],
) -> Image.Image:
    """Alpha blends a foreground RGBA image over a background RGBA image.

    Args:
        fg_image: Foreground image. Can be a file path (str), a PIL Image,
            or a NumPy ndarray.
        bg_image: Background image. Can be a file path (str), a PIL Image,
            or a NumPy ndarray.

    Returns:
        A PIL Image representing the alpha-blended result in RGBA mode.
    """
    if isinstance(fg_image, str):
        fg_image = Image.open(fg_image)
    elif isinstance(fg_image, np.ndarray):
        fg_image = Image.fromarray(fg_image)

    if isinstance(bg_image, str):
        bg_image = Image.open(bg_image)
    elif isinstance(bg_image, np.ndarray):
        bg_image = Image.fromarray(bg_image)

    if fg_image.size != bg_image.size:
        raise ValueError(
            f"Image sizes not match {fg_image.size} v.s. {bg_image.size}."
        )

    fg = fg_image.convert("RGBA")
    bg = bg_image.convert("RGBA")

    return Image.alpha_composite(bg, fg)


def check_object_edge_truncated(
    mask: np.ndarray, edge_threshold: int = 5
) -> bool:
    """Checks if a binary object mask is truncated at the image edges.

    Args:
        mask: A 2D binary NumPy array where nonzero values indicate the object region.
        edge_threshold: Number of pixels from each image edge to consider for truncation.
            Defaults to 5.

    Returns:
        True if the object is fully enclosed (not truncated).
        False if the object touches or crosses any image boundary.
    """
    top = mask[:edge_threshold, :].any()
    bottom = mask[-edge_threshold:, :].any()
    left = mask[:, :edge_threshold].any()
    right = mask[:, -edge_threshold:].any()

    return not (top or bottom or left or right)


if __name__ == "__main__":
    image_paths = [
        "outputs/layouts_sim/task_0000/images/pen.png",
        "outputs/layouts_sim/task_0000/images/notebook.png",
        "outputs/layouts_sim/task_0000/images/mug.png",
        "outputs/layouts_sim/task_0000/images/lamp.png",
        "outputs/layouts_sim2/task_0014/images/cloth.png",  # TODO
    ]
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = image[..., -1]
        flag = check_object_edge_truncated(mask)
        print(flag, image_path)
