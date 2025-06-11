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
import os
from typing import List, Literal, Union

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
import utils3d
import xatlas
from tqdm import tqdm
from embodied_gen.data.mesh_operator import MeshFixer
from embodied_gen.data.utils import (
    CameraSetting,
    get_images_from_grid,
    init_kal_camera,
    normalize_vertices_array,
    post_process_texture,
    save_mesh_with_mtl,
)
from embodied_gen.models.delight_model import DelightingModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


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
        images: List[np.array],
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


def kaolin_to_opencv_view(raw_matrix):
    R_orig = raw_matrix[:, :3, :3]
    t_orig = raw_matrix[:, :3, 3]

    R_target = torch.zeros_like(R_orig)
    R_target[:, :, 0] = R_orig[:, :, 2]
    R_target[:, :, 1] = R_orig[:, :, 0]
    R_target[:, :, 2] = R_orig[:, :, 1]

    t_target = t_orig

    target_matrix = (
        torch.eye(4, device=raw_matrix.device)
        .unsqueeze(0)
        .repeat(raw_matrix.size(0), 1, 1)
    )
    target_matrix[:, :3, :3] = R_target
    target_matrix[:, :3, 3] = t_target

    return target_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Render settings")

    parser.add_argument(
        "--mesh_path",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the mesh files for rendering.",
    )
    parser.add_argument(
        "--color_path",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the mesh files for rendering.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs",
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
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run on (default: `cuda`)",
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=1024,
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
        default=2500,
        help="Optimization steps for texture baking (default: 2500)",
    )
    parser.add_argument(
        "--mesh_sipmlify_ratio",
        type=float,
        default=0.9,
        help="Mesh simplification ratio (default: 0.9)",
    )
    parser.add_argument(
        "--no_coor_trans",
        action="store_true",
        help="Do not transform the asset coordinate system.",
    )
    parser.add_argument(
        "--delight", action="store_true", help="Use delighting model."
    )
    parser.add_argument(
        "--skip_fix_mesh", action="store_true", help="Fix mesh geometry."
    )

    args = parser.parse_args()

    if args.uuid is None:
        args.uuid = []
        for path in args.mesh_path:
            uuid = os.path.basename(path).split(".")[0]
            args.uuid.append(uuid)

    return args


def entrypoint() -> None:
    args = parse_args()
    camera_params = CameraSetting(
        num_images=args.num_images,
        elevation=args.elevation,
        distance=args.distance,
        resolution_hw=args.resolution_hw,
        fov=math.radians(args.fov),
        device=args.device,
    )

    for mesh_path, uuid, img_path in zip(
        args.mesh_path, args.uuid, args.color_path
    ):
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices, scale, center = normalize_vertices_array(mesh.vertices)

        if not args.no_coor_trans:
            x_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            z_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            vertices = vertices @ x_rot
            vertices = vertices @ z_rot

        faces = mesh.faces.astype(np.int32)
        vertices = vertices.astype(np.float32)

        if not args.skip_fix_mesh:
            mesh_fixer = MeshFixer(vertices, faces, args.device)
            vertices, faces = mesh_fixer(
                filter_ratio=args.mesh_sipmlify_ratio,
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
        images = get_images_from_grid(
            img_path, img_size=camera_params.resolution_hw[0]
        )
        if args.delight:
            delight_model = DelightingModel()
            images = [delight_model(img) for img in images]

        images = [np.array(img) for img in images]
        texture = texture_backer.bake_texture(
            images=[img[..., :3] for img in images],
            texture_size=args.texture_size,
            mode=args.baker_mode,
            opt_step=args.opt_step,
        )
        texture = post_process_texture(texture)

        if not args.no_coor_trans:
            vertices = vertices @ np.linalg.inv(z_rot)
            vertices = vertices @ np.linalg.inv(x_rot)
        vertices = vertices / scale
        vertices = vertices + center

        output_path = os.path.join(args.output_root, f"{uuid}.obj")
        mesh = save_mesh_with_mtl(vertices, faces, uvs, texture, output_path)

    return


if __name__ == "__main__":
    entrypoint()
