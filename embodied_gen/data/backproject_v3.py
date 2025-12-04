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
import logging
import math
from typing import Literal, Union

import cv2
import numpy as np
import nvdiffrast.torch as dr
import spaces
import torch
import trimesh
import utils3d
import xatlas
from PIL import Image
from tqdm import tqdm
from embodied_gen.data.mesh_operator import MeshFixer
from embodied_gen.data.utils import (
    CameraSetting,
    init_kal_camera,
    kaolin_to_opencv_view,
    normalize_vertices_array,
    post_process_texture,
    save_mesh_with_mtl,
)
from embodied_gen.models.delight_model import DelightingModel
from embodied_gen.models.gs_model import load_gs_model
from embodied_gen.models.sr_model import ImageRealESRGAN

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


__all__ = [
    "TextureBaker",
]


class TextureBaker(object):
    """Baking textures onto a mesh from multiple observations.

    This class take 3D mesh data, camera settings and texture baking parameters
    to generate texture map by projecting images to the mesh from diff views.
    It supports both a fast texture baking approach and a more optimized method
    with total variation regularization.

    Attributes:
        vertices (torch.Tensor): The vertices of the mesh.
        faces (torch.Tensor): The faces of the mesh, defined by vertex indices.
        uvs (torch.Tensor): The UV coordinates of the mesh.
        camera_params (CameraSetting): Camera setting (intrinsics, extrinsics).
        device (str): The device to run computations on ("cpu" or "cuda").
        w2cs (torch.Tensor): World-to-camera transformation matrices.
        projections (torch.Tensor): Camera projection matrices.

    Example:
        >>> vertices, faces, uvs = TextureBaker.parametrize_mesh(vertices, faces)  # noqa
        >>> texture_backer = TextureBaker(vertices, faces, uvs, camera_params)
        >>> images = get_images_from_grid(args.color_path, image_size)
        >>> texture = texture_backer.bake_texture(
        ...     images, texture_size=args.texture_size, mode=args.baker_mode
        ... )
        >>> texture = post_process_texture(texture)
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray,
        camera_params: CameraSetting,
        device: str = "cuda",
    ) -> None:
        self.vertices = (
            torch.tensor(vertices, device=device)
            if isinstance(vertices, np.ndarray)
            else vertices.to(device)
        )
        self.faces = (
            torch.tensor(faces.astype(np.int32), device=device)
            if isinstance(faces, np.ndarray)
            else faces.to(device)
        )
        self.uvs = (
            torch.tensor(uvs, device=device)
            if isinstance(uvs, np.ndarray)
            else uvs.to(device)
        )
        self.camera_params = camera_params
        self.device = device

        camera = init_kal_camera(camera_params)
        matrix_mv = camera.view_matrix()  # (n_cam 4 4) world2cam
        matrix_mv = kaolin_to_opencv_view(matrix_mv)
        matrix_p = (
            camera.intrinsics.projection_matrix()
        )  # (n_cam 4 4) cam2pixel
        self.w2cs = matrix_mv.to(self.device)
        self.projections = matrix_p.to(self.device)

    @staticmethod
    def parametrize_mesh(
        vertices: np.array, faces: np.array
    ) -> Union[np.array, np.array, np.array]:
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

        vertices = vertices[vmapping]
        faces = indices

        return vertices, faces, uvs

    def _bake_fast(self, observations, w2cs, projections, texture_size, masks):
        texture = torch.zeros(
            (texture_size * texture_size, 3), dtype=torch.float32
        ).cuda()
        texture_weights = torch.zeros(
            (texture_size * texture_size), dtype=torch.float32
        ).cuda()
        rastctx = utils3d.torch.RastContext(backend="cuda")
        for observation, w2c, projection in tqdm(
            zip(observations, w2cs, projections),
            total=len(observations),
            desc="Texture baking (fast)",
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    self.vertices[None],
                    self.faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=self.uvs[None],
                    view=w2c,
                    projection=projection,
                )
                uv_map = rast["uv"][0].detach().flip(0)
                mask = rast["mask"][0].detach().bool() & masks[0]

            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = (
                uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            )
            texture = texture.scatter_add(
                0, idx.view(-1, 1).expand(-1, 3), obs
            )
            texture_weights = texture_weights.scatter_add(
                0,
                idx,
                torch.ones(
                    (obs.shape[0]), dtype=torch.float32, device=texture.device
                ),
            )

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(
            texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255,
            0,
            255,
        ).astype(np.uint8)

        # inpaint
        mask = (
            (texture_weights == 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
            .reshape(texture_size, texture_size)
        )
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

        return texture

    def _bake_opt(
        self,
        observations,
        w2cs,
        projections,
        texture_size,
        lambda_tv,
        masks,
        total_steps,
    ):
        rastctx = utils3d.torch.RastContext(backend="cuda")
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        for observation, w2c, projection in tqdm(
            zip(observations, w2cs, projections),
            total=len(w2cs),
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    self.vertices[None],
                    self.faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=self.uvs[None],
                    view=w2c,
                    projection=projection,
                )
                _uv.append(rast["uv"].detach())
                _uv_dr.append(rast["uv_dr"].detach())

        texture = torch.nn.Parameter(
            torch.zeros(
                (1, texture_size, texture_size, 3), dtype=torch.float32
            ).cuda()
        )
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def cosine_anealing(step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (
                1 + np.cos(np.pi * step / total_steps)
            )

        def tv_loss(texture):
            return torch.nn.functional.l1_loss(
                texture[:, :-1, :, :], texture[:, 1:, :, :]
            ) + torch.nn.functional.l1_loss(
                texture[:, :, :-1, :], texture[:, :, 1:, :]
            )

        with tqdm(total=total_steps, desc="Texture baking") as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(w2cs))
                uv, uv_dr, observation, mask = (
                    _uv[selected],
                    _uv_dr[selected],
                    observations[selected],
                    masks[selected],
                )
                render = dr.texture(texture, uv, uv_dr)[0]
                loss = torch.nn.functional.l1_loss(
                    render[mask], observation[mask]
                )
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()

                optimizer.param_groups[0]["lr"] = cosine_anealing(
                    step, total_steps, 1e-2, 1e-5
                )
                pbar.set_postfix({"loss": loss.item()})
                pbar.update()

        texture = np.clip(
            texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx,
            (self.uvs * 2 - 1)[None],
            self.faces,
            texture_size,
            texture_size,
        )["mask"][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

        return texture

    def bake_texture(
        self,
        images: list[np.array],
        texture_size: int = 1024,
        mode: Literal["fast", "opt"] = "opt",
        lambda_tv: float = 1e-2,
        opt_step: int = 2000,
    ):
        masks = [np.any(img > 0, axis=-1) for img in images]
        masks = [torch.tensor(m > 0).bool().to(self.device) for m in masks]
        images = [
            torch.tensor(obs / 255.0).float().to(self.device) for obs in images
        ]

        if mode == "fast":
            return self._bake_fast(
                images, self.w2cs, self.projections, texture_size, masks
            )
        elif mode == "opt":
            return self._bake_opt(
                images,
                self.w2cs,
                self.projections,
                texture_size,
                lambda_tv,
                masks,
                opt_step,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    """Parses command-line arguments for texture backprojection.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Backproject texture")
    parser.add_argument(
        "--gs_path",
        type=str,
        help="Path to the GS.ply gaussian splatting model",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        help="Mesh path, .obj, .glb or .ply",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output mesh path with suffix",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=180,
        help="Number of images to render.",
    )
    parser.add_argument(
        "--elevation",
        nargs="+",
        type=float,
        default=list(range(85, -90, -10)),
        help="Elevation angles for the camera",
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
        help="Resolution of the render images (default: (512, 512))",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=30,
        help="Field of view in degrees (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run on (default: `cuda`)",
    )
    parser.add_argument(
        "--skip_fix_mesh", action="store_true", help="Fix mesh geometry."
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=2048,
        help="Texture size for texture baking (default: 1024)",
    )
    parser.add_argument(
        "--baker_mode",
        type=str,
        default="opt",
        help="Texture baking mode, `fast` or `opt` (default: opt)",
    )
    parser.add_argument(
        "--opt_step",
        type=int,
        default=3000,
        help="Optimization steps for texture baking (default: 3000)",
    )
    parser.add_argument(
        "--mesh_sipmlify_ratio",
        type=float,
        default=0.9,
        help="Mesh simplification ratio (default: 0.9)",
    )
    parser.add_argument(
        "--delight", action="store_true", help="Use delighting model."
    )
    parser.add_argument(
        "--no_smooth_texture",
        action="store_true",
        help="Do not smooth the texture.",
    )
    parser.add_argument(
        "--no_coor_trans",
        action="store_true",
        help="Do not transform the asset coordinate system.",
    )
    parser.add_argument(
        "--save_glb_path", type=str, default=None, help="Save glb path."
    )
    parser.add_argument("--n_max_faces", type=int, default=30000)
    args, unknown = parser.parse_known_args()

    return args


def entrypoint(
    delight_model: DelightingModel = None,
    imagesr_model: ImageRealESRGAN = None,
    **kwargs,
) -> trimesh.Trimesh:
    """Entrypoint for texture backprojection from multi-view images.

    Args:
        delight_model (DelightingModel, optional): Delighting model.
        imagesr_model (ImageRealESRGAN, optional): Super-resolution model.
        **kwargs: Additional arguments to override CLI.

    Returns:
        trimesh.Trimesh: Textured mesh.
    """
    args = parse_args()
    for k, v in kwargs.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    # Setup camera parameters.
    camera_params = CameraSetting(
        num_images=args.num_images,
        elevation=args.elevation,
        distance=args.distance,
        resolution_hw=args.resolution_hw,
        fov=math.radians(args.fov),
        device=args.device,
    )

    # GS render.
    camera = init_kal_camera(camera_params, flip_az=True)
    matrix_mv = camera.view_matrix()  # (n_cam 4 4) world2cam
    matrix_mv[:, :3, 3] = -matrix_mv[:, :3, 3]
    w2cs = matrix_mv.to(camera_params.device)
    c2ws = [torch.linalg.inv(matrix) for matrix in w2cs]
    Ks = torch.tensor(camera_params.Ks).to(camera_params.device)
    gs_model = load_gs_model(args.gs_path, pre_quat=[0.0, 0.0, 1.0, 0.0])
    multiviews = []
    for idx in tqdm(range(len(c2ws)), desc="Rendering GS"):
        result = gs_model.render(
            c2ws[idx],
            Ks=Ks,
            image_width=camera_params.resolution_hw[1],
            image_height=camera_params.resolution_hw[0],
        )
        color = cv2.cvtColor(result.rgba, cv2.COLOR_BGRA2RGBA)
        multiviews.append(Image.fromarray(color))

    if args.delight and delight_model is None:
        delight_model = DelightingModel()

    if args.delight:
        for idx in range(len(multiviews)):
            multiviews[idx] = delight_model(multiviews[idx])

    multiviews = [img.convert("RGB") for img in multiviews]

    mesh = trimesh.load(args.mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    vertices, scale, center = normalize_vertices_array(mesh.vertices)

    # Transform mesh coordinate system by default.
    if not args.no_coor_trans:
        x_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        z_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        vertices = vertices @ x_rot
        vertices = vertices @ z_rot

    faces = mesh.faces.astype(np.int32)
    vertices = vertices.astype(np.float32)

    if not args.skip_fix_mesh and len(faces) > 10 * args.n_max_faces:
        mesh_fixer = MeshFixer(vertices, faces, args.device)
        vertices, faces = mesh_fixer(
            filter_ratio=args.mesh_sipmlify_ratio,
            max_hole_size=0.04,
            resolution=1024,
            num_views=1000,
            norm_mesh_ratio=0.5,
        )
        if len(faces) > args.n_max_faces:
            mesh_fixer = MeshFixer(vertices, faces, args.device)
            vertices, faces = mesh_fixer(
                filter_ratio=max(0.05, args.mesh_sipmlify_ratio - 0.2),
                max_hole_size=0.04,
                resolution=1024,
                num_views=1000,
                norm_mesh_ratio=0.5,
            )

    vertices, faces, uvs = TextureBaker.parametrize_mesh(vertices, faces)
    texture_backer = TextureBaker(
        vertices,
        faces,
        uvs,
        camera_params,
    )

    multiviews = [np.array(img) for img in multiviews]
    texture = texture_backer.bake_texture(
        images=[img[..., :3] for img in multiviews],
        texture_size=args.texture_size,
        mode=args.baker_mode,
        opt_step=args.opt_step,
    )
    if not args.no_smooth_texture:
        texture = post_process_texture(texture)

    # Recover mesh original orientation, scale and center.
    if not args.no_coor_trans:
        vertices = vertices @ np.linalg.inv(z_rot)
        vertices = vertices @ np.linalg.inv(x_rot)
    vertices = vertices / scale
    vertices = vertices + center

    textured_mesh = save_mesh_with_mtl(
        vertices, faces, uvs, texture, args.output_path
    )
    if args.save_glb_path is not None:
        os.makedirs(os.path.dirname(args.save_glb_path), exist_ok=True)
        textured_mesh.export(args.save_glb_path)

    return textured_mesh


if __name__ == "__main__":
    entrypoint()
