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


import argparse
import json
import logging
import math
import os
from collections import defaultdict
from typing import List, Union

import cv2
import imageio
import numpy as np
import nvdiffrast.torch as dr
import PIL.Image as Image
import torch
from tqdm import tqdm
from embodied_gen.data.utils import (
    CameraSetting,
    DiffrastRender,
    RenderItems,
    as_list,
    calc_vertex_normals,
    import_kaolin_mesh,
    init_kal_camera,
    normalize_vertices_array,
    render_pbr,
    save_images,
)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.expanduser(
    "~/.cache/torch_extensions"
)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


__all__ = [
    "ImageRender",
    "create_mp4_from_images",
    "create_gif_from_images",
]


def create_mp4_from_images(
    images: list[np.ndarray],
    output_path: str,
    fps: int = 10,
    prompt: str = None,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 255, 255)
    position = (20, 25)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for image in images:
            image = image.clip(min=0, max=1)
            image = (255.0 * image).astype(np.uint8)
            image = image[..., :3]
            if prompt is not None:
                cv2.putText(
                    image,
                    prompt,
                    position,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                )

            writer.append_data(image)

    logger.info(f"MP4 video saved to {output_path}")


def create_gif_from_images(
    images: list[np.ndarray], output_path: str, fps: int = 10
) -> None:
    pil_images = []
    for image in images:
        image = image.clip(min=0, max=1)
        image = (255.0 * image).astype(np.uint8)
        image = Image.fromarray(image, mode="RGBA")
        pil_images.append(image.convert("RGB"))

    duration = 1000 // fps
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )

    logger.info(f"GIF saved to {output_path}")


class ImageRender(object):
    """A differentiable mesh renderer supporting multi-view rendering.

    This class wraps a differentiable rasterization using `nvdiffrast` to
    render mesh geometry to various maps (normal, depth, alpha, albedo, etc.).

    Args:
        render_items (list[RenderItems]): A list of rendering targets to
            generate (e.g., IMAGE, DEPTH, NORMAL, etc.).
        camera_params (CameraSetting): The camera parameters for rendering,
            including intrinsic and extrinsic matrices.
        recompute_vtx_normal (bool, optional): If True, recomputes
            vertex normals from the mesh geometry. Defaults to True.
        with_mtl (bool, optional): Whether to load `.mtl` material files
            for meshes. Defaults to False.
        gen_color_gif (bool, optional): Generate a GIF of rendered
            color images. Defaults to False.
        gen_color_mp4 (bool, optional): Generate an MP4 video of rendered
            color images. Defaults to False.
        gen_viewnormal_mp4 (bool, optional): Generate an MP4 video of
            view-space normals. Defaults to False.
        gen_glonormal_mp4 (bool, optional): Generate an MP4 video of
            global-space normals. Defaults to False.
        no_index_file (bool, optional): If True, skip saving the `index.json`
            summary file. Defaults to False.
        light_factor (float, optional): A scalar multiplier for
            PBR light intensity. Defaults to 1.0.
    """

    def __init__(
        self,
        render_items: list[RenderItems],
        camera_params: CameraSetting,
        recompute_vtx_normal: bool = True,
        with_mtl: bool = False,
        gen_color_gif: bool = False,
        gen_color_mp4: bool = False,
        gen_viewnormal_mp4: bool = False,
        gen_glonormal_mp4: bool = False,
        no_index_file: bool = False,
        light_factor: float = 1.0,
    ) -> None:
        camera = init_kal_camera(camera_params)
        self.camera = camera

        # Setup MVP matrix and renderer.
        mv = camera.view_matrix()  # (n 4 4) world2cam
        p = camera.intrinsics.projection_matrix()
        # NOTE: add a negative sign at P[0, 2] as the y axis is flipped in `nvdiffrast` output.  # noqa
        p[:, 1, 1] = -p[:, 1, 1]
        # mvp = torch.bmm(p, mv) # camera.view_projection_matrix()
        self.mv = mv
        self.p = p

        renderer = DiffrastRender(
            p_matrix=p,
            mv_matrix=mv,
            resolution_hw=camera_params.resolution_hw,
            context=dr.RasterizeCudaContext(),
            mask_thresh=0.5,
            grad_db=False,
            device=camera_params.device,
            antialias_mask=True,
        )
        self.renderer = renderer
        self.recompute_vtx_normal = recompute_vtx_normal
        self.render_items = render_items
        self.device = camera_params.device
        self.with_mtl = with_mtl
        self.gen_color_gif = gen_color_gif
        self.gen_color_mp4 = gen_color_mp4
        self.gen_viewnormal_mp4 = gen_viewnormal_mp4
        self.gen_glonormal_mp4 = gen_glonormal_mp4
        self.light_factor = light_factor
        self.no_index_file = no_index_file

    def render_mesh(
        self,
        mesh_path: Union[str, List[str]],
        output_root: str,
        uuid: Union[str, List[str]] = None,
        prompts: List[str] = None,
    ) -> None:
        mesh_path = as_list(mesh_path)
        if uuid is None:
            uuid = [os.path.basename(p).split(".")[0] for p in mesh_path]
        uuid = as_list(uuid)
        assert len(mesh_path) == len(uuid)
        os.makedirs(output_root, exist_ok=True)

        meta_info = dict()
        for idx, (path, uid) in tqdm(
            enumerate(zip(mesh_path, uuid)), total=len(mesh_path)
        ):
            output_dir = os.path.join(output_root, uid)
            os.makedirs(output_dir, exist_ok=True)
            prompt = prompts[idx] if prompts else None
            data_dict = self(path, output_dir, prompt)
            meta_info[uid] = data_dict

        if self.no_index_file:
            return

        index_file = os.path.join(output_root, "index.json")
        with open(index_file, "w") as fout:
            json.dump(meta_info, fout)

        logger.info(f"Rendering meta info logged in {index_file}")

    def __call__(
        self, mesh_path: str, output_dir: str, prompt: str = None
    ) -> dict[str, str]:
        """Render a single mesh and return paths to the rendered outputs.

        Processes the input mesh, renders multiple modalities (e.g., normals,
        depth, albedo), and optionally saves video or image sequences.

        Args:
            mesh_path (str): Path to the mesh file (.obj/.glb).
            output_dir (str): Directory to save rendered outputs.
            prompt (str, optional): Optional caption prompt for MP4 metadata.

        Returns:
            dict[str, str]: A mapping render types to the saved image paths.
        """
        try:
            mesh = import_kaolin_mesh(mesh_path, self.with_mtl)
        except Exception as e:
            logger.error(f"[ERROR MESH LOAD]: {e}, skip {mesh_path}")
            return

        mesh.vertices, scale, center = normalize_vertices_array(mesh.vertices)
        if self.recompute_vtx_normal:
            mesh.vertex_normals = calc_vertex_normals(
                mesh.vertices, mesh.faces
            )

        mesh = mesh.to(self.device)
        vertices, faces, vertex_normals = (
            mesh.vertices,
            mesh.faces,
            mesh.vertex_normals,
        )

        # Perform rendering.
        data_dict = defaultdict(list)
        if RenderItems.ALPHA.value in self.render_items:
            masks, _ = self.renderer.render_rast_alpha(vertices, faces)
            render_paths = save_images(
                masks, f"{output_dir}/{RenderItems.ALPHA}"
            )
            data_dict[RenderItems.ALPHA.value] = render_paths

        if RenderItems.GLOBAL_NORMAL.value in self.render_items:
            rendered_normals, masks = self.renderer.render_global_normal(
                vertices, faces, vertex_normals
            )
            if self.gen_glonormal_mp4:
                if isinstance(rendered_normals, torch.Tensor):
                    rendered_normals = rendered_normals.detach().cpu().numpy()
                create_mp4_from_images(
                    rendered_normals,
                    output_path=f"{output_dir}/normal.mp4",
                    fps=15,
                    prompt=prompt,
                )
            else:
                render_paths = save_images(
                    rendered_normals,
                    f"{output_dir}/{RenderItems.GLOBAL_NORMAL}",
                    cvt_color=cv2.COLOR_BGR2RGB,
                )
                data_dict[RenderItems.GLOBAL_NORMAL.value] = render_paths

            if RenderItems.VIEW_NORMAL.value in self.render_items:
                assert (
                    RenderItems.GLOBAL_NORMAL in self.render_items
                ), f"Must render global normal firstly, got render_items: {self.render_items}."  # noqa
                rendered_view_normals = self.renderer.transform_normal(
                    rendered_normals, self.mv, masks, to_view=True
                )

                if self.gen_viewnormal_mp4:
                    create_mp4_from_images(
                        rendered_view_normals,
                        output_path=f"{output_dir}/view_normal.mp4",
                        fps=15,
                        prompt=prompt,
                    )
                else:
                    render_paths = save_images(
                        rendered_view_normals,
                        f"{output_dir}/{RenderItems.VIEW_NORMAL}",
                        cvt_color=cv2.COLOR_BGR2RGB,
                    )
                    data_dict[RenderItems.VIEW_NORMAL.value] = render_paths

        if RenderItems.POSITION_MAP.value in self.render_items:
            rendered_position, masks = self.renderer.render_position(
                vertices, faces
            )
            norm_position = self.renderer.normalize_map_by_mask(
                rendered_position, masks
            )
            render_paths = save_images(
                norm_position,
                f"{output_dir}/{RenderItems.POSITION_MAP}",
                cvt_color=cv2.COLOR_BGR2RGB,
            )
            data_dict[RenderItems.POSITION_MAP.value] = render_paths

        if RenderItems.DEPTH.value in self.render_items:
            rendered_depth, masks = self.renderer.render_depth(vertices, faces)
            norm_depth = self.renderer.normalize_map_by_mask(
                rendered_depth, masks
            )
            render_paths = save_images(
                norm_depth,
                f"{output_dir}/{RenderItems.DEPTH}",
            )
            data_dict[RenderItems.DEPTH.value] = render_paths

            render_paths = save_images(
                rendered_depth,
                f"{output_dir}/{RenderItems.DEPTH}_exr",
                to_uint8=False,
                format=".exr",
            )
            data_dict[f"{RenderItems.DEPTH.value}_exr"] = render_paths

        if RenderItems.IMAGE.value in self.render_items:
            images = []
            albedos = []
            diffuses = []
            masks, _ = self.renderer.render_rast_alpha(vertices, faces)
            try:
                for idx, cam in enumerate(self.camera):
                    image, albedo, diffuse, _ = render_pbr(
                        mesh, cam, light_factor=self.light_factor
                    )
                    image = torch.cat([image[0], masks[idx]], axis=-1)
                    images.append(image.detach().cpu().numpy())

                    if RenderItems.ALBEDO.value in self.render_items:
                        albedo = torch.cat([albedo[0], masks[idx]], axis=-1)
                        albedos.append(albedo.detach().cpu().numpy())

                    if RenderItems.DIFFUSE.value in self.render_items:
                        diffuse = torch.cat([diffuse[0], masks[idx]], axis=-1)
                        diffuses.append(diffuse.detach().cpu().numpy())

            except Exception as e:
                logger.error(f"[ERROR pbr render]: {e}, skip {mesh_path}")
                return

            if self.gen_color_gif:
                create_gif_from_images(
                    images,
                    output_path=f"{output_dir}/color.gif",
                    fps=15,
                )

            if self.gen_color_mp4:
                create_mp4_from_images(
                    images,
                    output_path=f"{output_dir}/color.mp4",
                    fps=15,
                    prompt=prompt,
                )

            if self.gen_color_mp4 or self.gen_color_gif:
                return data_dict

            render_paths = save_images(
                images,
                f"{output_dir}/{RenderItems.IMAGE}",
                cvt_color=cv2.COLOR_BGRA2RGBA,
            )
            data_dict[RenderItems.IMAGE.value] = render_paths

            render_paths = save_images(
                albedos,
                f"{output_dir}/{RenderItems.ALBEDO}",
                cvt_color=cv2.COLOR_BGRA2RGBA,
            )
            data_dict[RenderItems.ALBEDO.value] = render_paths

            render_paths = save_images(
                diffuses,
                f"{output_dir}/{RenderItems.DIFFUSE}",
                cvt_color=cv2.COLOR_BGRA2RGBA,
            )
            data_dict[RenderItems.DIFFUSE.value] = render_paths

        data_dict["status"] = "success"

        logger.info(f"Finish rendering in {output_dir}")

        return data_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Render settings")

    parser.add_argument(
        "--mesh_path",
        type=str,
        nargs="+",
        help="Paths to the mesh files for rendering.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Root directory for output",
    )
    parser.add_argument(
        "--uuid",
        type=str,
        nargs="+",
        default=None,
        help="uuid for rendering saving.",
    )
    parser.add_argument(
        "--num_images", type=int, default=6, help="Number of images to render."
    )
    parser.add_argument(
        "--elevation",
        type=float,
        nargs="+",
        default=[20.0, -10.0],
        help="Elevation angles for the camera (default: [20.0, -10.0])",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=5,
        help="Camera distance (default: 5)",
    )
    parser.add_argument(
        "--resolution_hw",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Resolution of the output images (default: (512, 512))",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=30,
        help="Field of view in degrees (default: 30)",
    )
    parser.add_argument(
        "--pbr_light_factor",
        type=float,
        default=1.0,
        help="Light factor for mesh PBR rendering (default: 2.)",
    )
    parser.add_argument(
        "--with_mtl",
        action="store_true",
        help="Whether to render with mesh material.",
    )
    parser.add_argument(
        "--gen_color_gif",
        action="store_true",
        help="Whether to generate color .gif rendering file.",
    )
    parser.add_argument(
        "--gen_color_mp4",
        action="store_true",
        help="Whether to generate color .mp4 rendering file.",
    )
    parser.add_argument(
        "--gen_viewnormal_mp4",
        action="store_true",
        help="Whether to generate view normal .mp4 rendering file.",
    )
    parser.add_argument(
        "--gen_glonormal_mp4",
        action="store_true",
        help="Whether to generate global normal .mp4 rendering file.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Text prompts for the rendering.",
    )

    args, unknown = parser.parse_known_args()

    if args.uuid is None and args.mesh_path is not None:
        args.uuid = []
        for path in args.mesh_path:
            uuid = os.path.basename(path).split(".")[0]
            args.uuid.append(uuid)

    return args


def entrypoint(**kwargs) -> None:
    args = parse_args()
    for k, v in kwargs.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    camera_settings = CameraSetting(
        num_images=args.num_images,
        elevation=args.elevation,
        distance=args.distance,
        resolution_hw=args.resolution_hw,
        fov=math.radians(args.fov),
        device="cuda",
    )

    render_items = [
        RenderItems.ALPHA.value,
        RenderItems.GLOBAL_NORMAL.value,
        RenderItems.VIEW_NORMAL.value,
        RenderItems.POSITION_MAP.value,
        RenderItems.IMAGE.value,
        RenderItems.DEPTH.value,
        # RenderItems.ALBEDO.value,
        # RenderItems.DIFFUSE.value,
    ]

    gen_video = (
        args.gen_color_gif
        or args.gen_color_mp4
        or args.gen_viewnormal_mp4
        or args.gen_glonormal_mp4
    )
    if gen_video:
        render_items = []
        if args.gen_color_gif or args.gen_color_mp4:
            render_items.append(RenderItems.IMAGE.value)
        if args.gen_glonormal_mp4:
            render_items.append(RenderItems.GLOBAL_NORMAL.value)
        if args.gen_viewnormal_mp4:
            render_items.append(RenderItems.VIEW_NORMAL.value)
            if RenderItems.GLOBAL_NORMAL.value not in render_items:
                render_items.append(RenderItems.GLOBAL_NORMAL.value)

    image_render = ImageRender(
        render_items=render_items,
        camera_params=camera_settings,
        with_mtl=args.with_mtl,
        gen_color_gif=args.gen_color_gif,
        gen_color_mp4=args.gen_color_mp4,
        gen_viewnormal_mp4=args.gen_viewnormal_mp4,
        gen_glonormal_mp4=args.gen_glonormal_mp4,
        light_factor=args.pbr_light_factor,
        no_index_file=gen_video,
    )
    image_render.render_mesh(
        mesh_path=args.mesh_path,
        output_root=args.output_root,
        uuid=args.uuid,
        prompts=args.prompts,
    )

    return


if __name__ == "__main__":
    entrypoint()
