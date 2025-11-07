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

import cv2
import numpy as np
import nvdiffrast.torch as dr
import spaces
import torch
import torch.nn.functional as F
import trimesh
import xatlas
from PIL import Image
from embodied_gen.data.mesh_operator import MeshFixer
from embodied_gen.data.utils import (
    CameraSetting,
    DiffrastRender,
    as_list,
    get_images_from_grid,
    init_kal_camera,
    normalize_vertices_array,
    post_process_texture,
    save_mesh_with_mtl,
)
from embodied_gen.models.delight_model import DelightingModel
from embodied_gen.models.sr_model import ImageRealESRGAN
from embodied_gen.utils.process_media import vcat_pil_images

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


__all__ = [
    "TextureBacker",
]


def _transform_vertices(
    mtx: torch.Tensor, pos: torch.Tensor, keepdim: bool = False
) -> torch.Tensor:
    """Transforms 3D vertices using a projection matrix.

    Args:
        mtx (torch.Tensor): Projection matrix.
        pos (torch.Tensor): Vertex positions.
        keepdim (bool, optional): If True, keeps the batch dimension.

    Returns:
        torch.Tensor: Transformed vertices.
    """
    t_mtx = torch.as_tensor(mtx, device=pos.device, dtype=pos.dtype)
    if pos.size(-1) == 3:
        pos = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)

    result = pos @ t_mtx.T

    return result if keepdim else result.unsqueeze(0)


def _bilinear_interpolation_scattering(
    image_h: int, image_w: int, coords: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Performs bilinear interpolation scattering for grid-based value accumulation.

    Args:
        image_h (int): Image height.
        image_w (int): Image width.
        coords (torch.Tensor): Normalized coordinates.
        values (torch.Tensor): Values to scatter.

    Returns:
        torch.Tensor: Interpolated grid.
    """
    device = values.device
    dtype = values.dtype
    C = values.shape[-1]

    indices = coords * torch.tensor(
        [image_h - 1, image_w - 1], dtype=dtype, device=device
    )
    i, j = indices.unbind(-1)

    i0, j0 = (
        indices.floor()
        .long()
        .clamp(0, image_h - 2)
        .clamp(0, image_w - 2)
        .unbind(-1)
    )
    i1, j1 = i0 + 1, j0 + 1

    w_i = i - i0.float()
    w_j = j - j0.float()
    weights = torch.stack(
        [(1 - w_i) * (1 - w_j), (1 - w_i) * w_j, w_i * (1 - w_j), w_i * w_j],
        dim=1,
    )

    indices_comb = torch.stack(
        [
            torch.stack([i0, j0], dim=1),
            torch.stack([i0, j1], dim=1),
            torch.stack([i1, j0], dim=1),
            torch.stack([i1, j1], dim=1),
        ],
        dim=1,
    )

    grid = torch.zeros(image_h, image_w, C, device=device, dtype=dtype)
    cnt = torch.zeros(image_h, image_w, 1, device=device, dtype=dtype)

    for k in range(4):
        idx = indices_comb[:, k]
        w = weights[:, k].unsqueeze(-1)

        stride = torch.tensor([image_w, 1], device=device, dtype=torch.long)
        flat_idx = (idx * stride).sum(-1)

        grid.view(-1, C).scatter_add_(
            0, flat_idx.unsqueeze(-1).expand(-1, C), values * w
        )
        cnt.view(-1, 1).scatter_add_(0, flat_idx.unsqueeze(-1), w)

    mask = cnt.squeeze(-1) > 0
    grid[mask] = grid[mask] / cnt[mask].repeat(1, C)

    return grid


def _texture_inpaint_smooth(
    texture: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Performs texture inpainting using vertex-based color propagation.

    Args:
        texture (np.ndarray): Texture image.
        mask (np.ndarray): Mask image.
        vertices (np.ndarray): Mesh vertices.
        faces (np.ndarray): Mesh faces.
        uv_map (np.ndarray): UV coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: Inpainted texture and updated mask.
    """
    image_h, image_w, C = texture.shape
    N = vertices.shape[0]

    # Initialize vertex data structures
    vtx_mask = np.zeros(N, dtype=np.float32)
    vtx_colors = np.zeros((N, C), dtype=np.float32)
    unprocessed = []
    adjacency = [[] for _ in range(N)]

    # Build adjacency graph and initial color assignment
    for face_idx in range(faces.shape[0]):
        for k in range(3):
            uv_idx_k = faces[face_idx, k]
            v_idx = faces[face_idx, k]

            # Convert UV to pixel coordinates with boundary clamping
            u = np.clip(
                int(round(uv_map[uv_idx_k, 0] * (image_w - 1))), 0, image_w - 1
            )
            v = np.clip(
                int(round((1.0 - uv_map[uv_idx_k, 1]) * (image_h - 1))),
                0,
                image_h - 1,
            )

            if mask[v, u]:
                vtx_mask[v_idx] = 1.0
                vtx_colors[v_idx] = texture[v, u]
            elif v_idx not in unprocessed:
                unprocessed.append(v_idx)

            # Build undirected adjacency graph
            neighbor = faces[face_idx, (k + 1) % 3]
            if neighbor not in adjacency[v_idx]:
                adjacency[v_idx].append(neighbor)
            if v_idx not in adjacency[neighbor]:
                adjacency[neighbor].append(v_idx)

    # Color propagation with dynamic stopping
    remaining_iters, prev_count = 2, 0
    while remaining_iters > 0:
        current_unprocessed = []

        for v_idx in unprocessed:
            valid_neighbors = [n for n in adjacency[v_idx] if vtx_mask[n] > 0]
            if not valid_neighbors:
                current_unprocessed.append(v_idx)
                continue

            # Calculate inverse square distance weights
            neighbors_pos = vertices[valid_neighbors]
            dist_sq = np.sum((vertices[v_idx] - neighbors_pos) ** 2, axis=1)
            weights = 1 / np.maximum(dist_sq, 1e-8)

            vtx_colors[v_idx] = np.average(
                vtx_colors[valid_neighbors], weights=weights, axis=0
            )
            vtx_mask[v_idx] = 1.0

        # Update iteration control
        if len(current_unprocessed) == prev_count:
            remaining_iters -= 1
        else:
            remaining_iters = min(remaining_iters + 1, 2)
        prev_count = len(current_unprocessed)
        unprocessed = current_unprocessed

    # Generate output texture
    inpainted_texture, updated_mask = texture.copy(), mask.copy()
    for face_idx in range(faces.shape[0]):
        for k in range(3):
            v_idx = faces[face_idx, k]
            if not vtx_mask[v_idx]:
                continue

            # UV coordinate conversion
            uv_idx_k = faces[face_idx, k]
            u = np.clip(
                int(round(uv_map[uv_idx_k, 0] * (image_w - 1))), 0, image_w - 1
            )
            v = np.clip(
                int(round((1.0 - uv_map[uv_idx_k, 1]) * (image_h - 1))),
                0,
                image_h - 1,
            )

            inpainted_texture[v, u] = vtx_colors[v_idx]
            updated_mask[v, u] = 255

    return inpainted_texture, updated_mask


class TextureBacker:
    """Texture baking pipeline for multi-view projection and fusion.

    This class generates UV-based textures for a 3D mesh using multi-view images,
    depth, and normal information. It includes mesh normalization, UV unwrapping,
    visibility-aware back-projection, confidence-weighted fusion, and inpainting.

    Args:
        camera_params (CameraSetting): Camera intrinsics and extrinsics.
        view_weights (list[float]): Weights for each view in texture fusion.
        render_wh (tuple[int, int], optional): Intermediate rendering resolution.
        texture_wh (tuple[int, int], optional): Output texture resolution.
        bake_angle_thresh (int, optional): Max angle for valid projection.
        mask_thresh (float, optional): Threshold for visibility masks.
        smooth_texture (bool, optional): Apply post-processing to texture.
        inpaint_smooth (bool, optional): Apply inpainting smoothing.

    Example:
        ```py
        from embodied_gen.data.backproject_v2 import TextureBacker
        from embodied_gen.data.utils import CameraSetting
        import trimesh
        from PIL import Image

        camera_params = CameraSetting(
            num_images=6,
            elevation=[20, -10],
            distance=5,
            resolution_hw=(2048,2048),
            fov=math.radians(30),
            device='cuda',
        )
        view_weights = [1, 0.1, 0.02, 0.1, 1, 0.02]
        mesh = trimesh.load('mesh.obj')
        images = [Image.open(f'view_{i}.png') for i in range(6)]
        texture_backer = TextureBacker(camera_params, view_weights)
        textured_mesh = texture_backer(images, mesh, 'output.obj')
        ```
    """

    def __init__(
        self,
        camera_params: CameraSetting,
        view_weights: list[float],
        render_wh: tuple[int, int] = (2048, 2048),
        texture_wh: tuple[int, int] = (2048, 2048),
        bake_angle_thresh: int = 75,
        mask_thresh: float = 0.5,
        smooth_texture: bool = True,
        inpaint_smooth: bool = False,
    ) -> None:
        self.camera_params = camera_params
        self.renderer = None
        self.view_weights = view_weights
        self.device = camera_params.device
        self.render_wh = render_wh
        self.texture_wh = texture_wh
        self.mask_thresh = mask_thresh
        self.smooth_texture = smooth_texture
        self.inpaint_smooth = inpaint_smooth

        self.bake_angle_thresh = bake_angle_thresh
        self.bake_unreliable_kernel_size = int(
            (2 / 512) * max(self.render_wh[0], self.render_wh[1])
        )

    def _lazy_init_render(self, camera_params, mask_thresh):
        """Lazily initializes the renderer.

        Args:
            camera_params (CameraSetting): Camera settings.
            mask_thresh (float): Mask threshold.
        """
        if self.renderer is None:
            camera = init_kal_camera(camera_params)
            mv = camera.view_matrix()  # (n 4 4) world2cam
            p = camera.intrinsics.projection_matrix()
            # NOTE: add a negative sign at P[0, 2] as the y axis is flipped in `nvdiffrast` output.  # noqa
            p[:, 1, 1] = -p[:, 1, 1]
            self.renderer = DiffrastRender(
                p_matrix=p,
                mv_matrix=mv,
                resolution_hw=camera_params.resolution_hw,
                context=dr.RasterizeCudaContext(),
                mask_thresh=mask_thresh,
                grad_db=False,
                device=self.device,
                antialias_mask=True,
            )

    def load_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Normalizes mesh and unwraps UVs.

        Args:
            mesh (trimesh.Trimesh): Input mesh.

        Returns:
            trimesh.Trimesh: Mesh with normalized vertices and UVs.
        """
        mesh.vertices, scale, center = normalize_vertices_array(mesh.vertices)
        self.scale, self.center = scale, center

        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        uvs[:, 1] = 1 - uvs[:, 1]
        mesh.vertices = mesh.vertices[vmapping]
        mesh.faces = indices
        mesh.visual.uv = uvs

        return mesh

    def get_mesh_np_attrs(
        self,
        mesh: trimesh.Trimesh,
        scale: float = None,
        center: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets mesh attributes as numpy arrays.

        Args:
            mesh (trimesh.Trimesh): Input mesh.
            scale (float, optional): Scale factor.
            center (np.ndarray, optional): Center offset.

        Returns:
            tuple: (vertices, faces, uv_map)
        """
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        uv_map = mesh.visual.uv.copy()
        uv_map[:, 1] = 1.0 - uv_map[:, 1]

        if scale is not None:
            vertices = vertices / scale
        if center is not None:
            vertices = vertices + center

        return vertices, faces, uv_map

    def _render_depth_edges(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Computes edge image from depth map.

        Args:
            depth_image (torch.Tensor): Depth map.

        Returns:
            torch.Tensor: Edge image.
        """
        depth_image_np = depth_image.cpu().numpy()
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)
        sketch_image = (
            torch.from_numpy(depth_edges).to(depth_image.device).float() / 255
        )
        sketch_image = sketch_image.unsqueeze(-1)

        return sketch_image

    def compute_enhanced_viewnormal(
        self, mv_mtx: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        """Computes enhanced view normals for mesh faces.

        Args:
            mv_mtx (torch.Tensor): View matrices.
            vertices (torch.Tensor): Mesh vertices.
            faces (torch.Tensor): Mesh faces.

        Returns:
            torch.Tensor: View normals.
        """
        rast, _ = self.renderer.compute_dr_raster(vertices, faces)
        rendered_view_normals = []
        for idx in range(len(mv_mtx)):
            pos_cam = _transform_vertices(mv_mtx[idx], vertices, keepdim=True)
            pos_cam = pos_cam[:, :3] / pos_cam[:, 3:]
            v0, v1, v2 = (pos_cam[faces[:, i]] for i in range(3))
            face_norm = F.normalize(
                torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1
            )
            vertex_norm = (
                torch.from_numpy(
                    trimesh.geometry.mean_vertex_normals(
                        len(pos_cam), faces.cpu(), face_norm.cpu()
                    )
                )
                .to(vertices.device)
                .contiguous()
            )
            im_base_normals, _ = dr.interpolate(
                vertex_norm[None, ...].float(),
                rast[idx : idx + 1],
                faces.to(torch.int32),
            )
            rendered_view_normals.append(im_base_normals)

        rendered_view_normals = torch.cat(rendered_view_normals, dim=0)

        return rendered_view_normals

    def back_project(
        self, image, vis_mask, depth, normal, uv
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-projects image and confidence to UV texture space.

        Args:
            image (PIL.Image or np.ndarray): Input image.
            vis_mask (torch.Tensor): Visibility mask.
            depth (torch.Tensor): Depth map.
            normal (torch.Tensor): Normal map.
            uv (torch.Tensor): UV coordinates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Texture and confidence map.
        """
        image = np.array(image)
        image = torch.as_tensor(image, device=self.device, dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        image = image / 255

        depth_inv = (1.0 - depth) * vis_mask
        sketch_image = self._render_depth_edges(depth_inv)

        cos = F.cosine_similarity(
            torch.tensor([[0, 0, 1]], device=self.device),
            normal.view(-1, 3),
        ).view_as(normal[..., :1])
        cos[cos < np.cos(np.radians(self.bake_angle_thresh))] = 0

        k = self.bake_unreliable_kernel_size * 2 + 1
        kernel = torch.ones((1, 1, k, k), device=self.device)

        vis_mask = vis_mask.permute(2, 0, 1).unsqueeze(0).float()
        vis_mask = F.conv2d(
            1.0 - vis_mask,
            kernel,
            padding=k // 2,
        )
        vis_mask = 1.0 - (vis_mask > 0).float()
        vis_mask = vis_mask.squeeze(0).permute(1, 2, 0)

        sketch_image = sketch_image.permute(2, 0, 1).unsqueeze(0)
        sketch_image = F.conv2d(sketch_image, kernel, padding=k // 2)
        sketch_image = (sketch_image > 0).float()
        sketch_image = sketch_image.squeeze(0).permute(1, 2, 0)
        vis_mask = vis_mask * (sketch_image < 0.5)

        cos[vis_mask == 0] = 0
        valid_pixels = (vis_mask != 0).view(-1)

        return (
            self._scatter_texture(uv, image, valid_pixels),
            self._scatter_texture(uv, cos, valid_pixels),
        )

    def _scatter_texture(self, uv, data, mask):
        """Scatters data to texture using UV coordinates and mask.

        Args:
            uv (torch.Tensor): UV coordinates.
            data (torch.Tensor): Data to scatter.
            mask (torch.Tensor): Mask for valid pixels.

        Returns:
            torch.Tensor: Scattered texture.
        """

        def __filter_data(data, mask):
            return data.view(-1, data.shape[-1])[mask]

        return _bilinear_interpolation_scattering(
            self.texture_wh[1],
            self.texture_wh[0],
            __filter_data(uv, mask)[..., [1, 0]],
            __filter_data(data, mask),
        )

    @torch.no_grad()
    def fast_bake_texture(
        self, textures: list[torch.Tensor], confidence_maps: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuses multiple textures and confidence maps.

        Args:
            textures (list[torch.Tensor]): List of textures.
            confidence_maps (list[torch.Tensor]): List of confidence maps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Fused texture and mask.
        """
        channel = textures[0].shape[-1]
        texture_merge = torch.zeros(self.texture_wh + [channel]).to(
            self.device
        )
        trust_map_merge = torch.zeros(self.texture_wh + [1]).to(self.device)
        for texture, cos_map in zip(textures, confidence_maps):
            view_sum = (cos_map > 0).sum()
            painted_sum = ((cos_map > 0) * (trust_map_merge > 0)).sum()
            if painted_sum / view_sum > 0.99:
                continue
            texture_merge += texture * cos_map
            trust_map_merge += cos_map
        texture_merge = texture_merge / torch.clamp(trust_map_merge, min=1e-8)

        return texture_merge, trust_map_merge > 1e-8

    def uv_inpaint(
        self, mesh: trimesh.Trimesh, texture: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Inpaints missing regions in the UV texture.

        Args:
            mesh (trimesh.Trimesh): Mesh.
            texture (np.ndarray): Texture image.
            mask (np.ndarray): Mask image.

        Returns:
            np.ndarray: Inpainted texture.
        """
        if self.inpaint_smooth:
            vertices, faces, uv_map = self.get_mesh_np_attrs(mesh)
            texture, mask = _texture_inpaint_smooth(
                texture, mask, vertices, faces, uv_map
            )

        texture = texture.clip(0, 1)
        texture = cv2.inpaint(
            (texture * 255).astype(np.uint8),
            255 - mask,
            3,
            cv2.INPAINT_NS,
        )

        return texture

    @spaces.GPU
    def compute_texture(
        self,
        colors: list[Image.Image],
        mesh: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        """Computes the fused texture for the mesh from multi-view images.

        Args:
            colors (list[Image.Image]): List of view images.
            mesh (trimesh.Trimesh): Mesh to texture.

        Returns:
            tuple[np.ndarray, np.ndarray]: Texture and mask.
        """
        self._lazy_init_render(self.camera_params, self.mask_thresh)

        vertices = torch.from_numpy(mesh.vertices).to(self.device).float()
        faces = torch.from_numpy(mesh.faces).to(self.device).to(torch.int)
        uv_map = torch.from_numpy(mesh.visual.uv).to(self.device).float()

        rendered_depth, masks = self.renderer.render_depth(vertices, faces)
        norm_deps = self.renderer.normalize_map_by_mask(rendered_depth, masks)
        render_uvs, _ = self.renderer.render_uv(vertices, faces, uv_map)
        view_normals = self.compute_enhanced_viewnormal(
            self.renderer.mv_mtx, vertices, faces
        )

        textures, weighted_cos_maps = [], []
        for color, mask, dep, normal, uv, weight in zip(
            colors,
            masks,
            norm_deps,
            view_normals,
            render_uvs,
            self.view_weights,
        ):
            texture, cos_map = self.back_project(color, mask, dep, normal, uv)
            textures.append(texture)
            weighted_cos_maps.append(weight * (cos_map**4))

        texture, mask = self.fast_bake_texture(textures, weighted_cos_maps)

        texture_np = texture.cpu().numpy()
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        return texture_np, mask_np

    def __call__(
        self,
        colors: list[Image.Image],
        mesh: trimesh.Trimesh,
        output_path: str,
    ) -> trimesh.Trimesh:
        """Runs the texture baking and exports the textured mesh.

        Args:
            colors (list[Image.Image]): List of input view images.
            mesh (trimesh.Trimesh): Input mesh to be textured.
            output_path (str): Path to save the output textured mesh.

        Returns:
            trimesh.Trimesh: The textured mesh with UV and texture image.
        """
        mesh = self.load_mesh(mesh)
        texture_np, mask_np = self.compute_texture(colors, mesh)

        texture_np = self.uv_inpaint(mesh, texture_np, mask_np)
        if self.smooth_texture:
            texture_np = post_process_texture(texture_np)

        vertices, faces, uv_map = self.get_mesh_np_attrs(
            mesh, self.scale, self.center
        )
        textured_mesh = save_mesh_with_mtl(
            vertices, faces, uv_map, texture_np, output_path
        )

        return textured_mesh


def parse_args():
    """Parses command-line arguments for texture backprojection.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Backproject texture")
    parser.add_argument(
        "--color_path",
        nargs="+",
        type=str,
        help="Multiview color image in grid file paths",
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
        "--num_images", type=int, default=6, help="Number of images to render."
    )
    parser.add_argument(
        "--elevation",
        nargs="+",
        type=float,
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
        default=(2048, 2048),
        help="Resolution of the output images (default: (2048, 2048))",
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
        "--texture_wh",
        nargs=2,
        type=int,
        default=[2048, 2048],
        help="Texture resolution width and height",
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
        "--save_glb_path", type=str, default=None, help="Save glb path."
    )
    parser.add_argument(
        "--no_save_delight_img",
        action="store_true",
        help="Disable saving delight image",
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

    args.color_path = as_list(args.color_path)
    if args.delight and delight_model is None:
        delight_model = DelightingModel()

    color_grid = [Image.open(color_path) for color_path in args.color_path]
    color_grid = vcat_pil_images(color_grid, image_mode="RGBA")
    if args.delight:
        color_grid = delight_model(color_grid)
        if not args.no_save_delight_img:
            save_dir = os.path.dirname(args.output_path)
            os.makedirs(save_dir, exist_ok=True)
            color_grid.save(f"{save_dir}/color_delight.png")

    multiviews = get_images_from_grid(color_grid, img_size=512)
    view_weights = [1, 0.1, 0.02, 0.1, 1, 0.02]
    view_weights += [0.01] * (len(multiviews) - len(view_weights))

    # Use RealESRGAN_x4plus for x4 (512->2048) image super resolution.
    if imagesr_model is None:
        imagesr_model = ImageRealESRGAN(outscale=4)
    multiviews = [imagesr_model(img) for img in multiviews]
    multiviews = [img.convert("RGB") for img in multiviews]
    mesh = trimesh.load(args.mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if not args.skip_fix_mesh:
        mesh.vertices, scale, center = normalize_vertices_array(mesh.vertices)
        mesh_fixer = MeshFixer(mesh.vertices, mesh.faces, args.device)
        mesh.vertices, mesh.faces = mesh_fixer(
            filter_ratio=args.mesh_sipmlify_ratio,
            max_hole_size=0.04,
            resolution=1024,
            num_views=1000,
            norm_mesh_ratio=0.5,
        )
        if len(mesh.faces) > args.n_max_faces:
            mesh.vertices, mesh.faces = mesh_fixer(
                filter_ratio=0.8,
                max_hole_size=0.04,
                resolution=1024,
                num_views=1000,
                norm_mesh_ratio=0.5,
            )
        # Restore scale.
        mesh.vertices = mesh.vertices / scale
        mesh.vertices = mesh.vertices + center

    # Baking texture to mesh.
    texture_backer = TextureBacker(
        camera_params=camera_params,
        view_weights=view_weights,
        render_wh=args.resolution_hw,
        texture_wh=args.texture_wh,
        smooth_texture=not args.no_smooth_texture,
    )

    textured_mesh = texture_backer(multiviews, mesh, args.output_path)

    if args.save_glb_path is not None:
        os.makedirs(os.path.dirname(args.save_glb_path), exist_ok=True)
        textured_mesh.export(args.save_glb_path)

    return textured_mesh


if __name__ == "__main__":
    entrypoint()
