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


import math
import os
import random
import zipfile
from shutil import rmtree
from typing import List, Tuple, Union

import cv2
import kaolin as kal
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from kolors.models.modeling_chatglm import ChatGLMModel
    from kolors.models.tokenization_chatglm import ChatGLMTokenizer
except ImportError:
    ChatGLMTokenizer = None
    ChatGLMModel = None
import logging
from dataclasses import dataclass, field

import trimesh
from kaolin.render.camera import Camera
from torch import nn

logger = logging.getLogger(__name__)


__all__ = [
    "DiffrastRender",
    "save_images",
    "render_pbr",
    "prelabel_text_feature",
    "calc_vertex_normals",
    "normalize_vertices_array",
    "load_mesh_to_unit_cube",
    "as_list",
    "CameraSetting",
    "import_kaolin_mesh",
    "save_mesh_with_mtl",
    "get_images_from_grid",
    "post_process_texture",
    "quat_mult",
    "quat_to_rotmat",
    "gamma_shs",
    "resize_pil",
    "trellis_preprocess",
    "delete_dir",
]


class DiffrastRender(object):
    """A class to handle differentiable rendering using nvdiffrast.

    This class provides methods to render position, depth, and normal maps
    with optional anti-aliasing and gradient disabling for rasterization.

    Attributes:
        p_mtx (torch.Tensor): Projection matrix.
        mv_mtx (torch.Tensor): Model-view matrix.
        mvp_mtx (torch.Tensor): Model-view-projection matrix, calculated as
            p_mtx @ mv_mtx if not provided.
        resolution_hw (Tuple[int, int]): Height and width of the rendering resolution.  # noqa
        _ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): Rasterization context.  # noqa
        mask_thresh (float): Threshold for mask creation.
        grad_db (bool): Whether to disable gradients during rasterization.
        antialias_mask (bool): Whether to apply anti-aliasing to the mask.
        device (str): Device used for rendering ('cuda' or 'cpu').
    """

    def __init__(
        self,
        p_matrix: torch.Tensor,
        mv_matrix: torch.Tensor,
        resolution_hw: Tuple[int, int],
        context: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = None,
        mvp_matrix: torch.Tensor = None,
        mask_thresh: float = 0.5,
        grad_db: bool = False,
        antialias_mask: bool = True,
        align_coordinate: bool = True,
        device: str = "cuda",
    ) -> None:
        self.p_mtx = p_matrix
        self.mv_mtx = mv_matrix
        if mvp_matrix is None:
            self.mvp_mtx = torch.bmm(p_matrix, mv_matrix)

        self.resolution_hw = resolution_hw
        if context is None:
            context = dr.RasterizeCudaContext(device=device)
        self._ctx = context
        self.mask_thresh = mask_thresh
        self.grad_db = grad_db
        self.antialias_mask = antialias_mask
        self.align_coordinate = align_coordinate
        self.device = device

    def compute_dr_raster(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices_clip = self.transform_vertices(vertices, matrix=self.mvp_mtx)
        rast, _ = dr.rasterize(
            self._ctx,
            vertices_clip,
            faces.int(),
            resolution=self.resolution_hw,
            grad_db=self.grad_db,
        )

        return rast, vertices_clip

    def transform_vertices(
        self,
        vertices: torch.Tensor,
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        verts_ones = torch.ones(
            (len(vertices), 1), device=vertices.device, dtype=vertices.dtype
        )
        verts_homo = torch.cat([vertices, verts_ones], dim=-1)
        trans_vertices = torch.matmul(verts_homo, matrix.permute(0, 2, 1))

        return trans_vertices

    def normalize_map_by_mask_separately(
        self, map: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Normalize each map separately by mask, normalized map in [0, 1].
        normalized_maps = []
        for map_item, mask_item in zip(map, mask):
            normalized_map = self.normalize_map_by_mask(map_item, mask_item)
            normalized_maps.append(normalized_map)

        normalized_maps = torch.stack(normalized_maps, dim=0)

        return normalized_maps

    @staticmethod
    def normalize_map_by_mask(
        map: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Normalize all maps in total by mask, normalized map in [0, 1].
        foreground = (mask == 1).squeeze(dim=-1)
        foreground_elements = map[foreground]
        if len(foreground_elements) == 0:
            return map

        min_val, _ = foreground_elements.min(dim=0)
        max_val, _ = foreground_elements.max(dim=0)
        val_range = (max_val - min_val).clip(min=1e-6)

        normalized_map = (map - min_val) / val_range
        normalized_map = torch.lerp(
            torch.zeros_like(normalized_map), normalized_map, mask
        )
        normalized_map[normalized_map < 0] = 0

        return normalized_map

    def _compute_mask(
        self,
        rast: torch.Tensor,
        vertices_clip: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        mask = (rast[..., 3:] > 0).float()
        mask = mask.clip(min=0, max=1)

        if self.antialias_mask is True:
            mask = dr.antialias(mask, rast, vertices_clip, faces)
        else:
            foreground = mask > self.mask_thresh
            mask[foreground] = 1
            mask[~foreground] = 0

        return mask

    def render_rast_alpha(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ):
        faces = faces.to(torch.int32)
        rast, vertices_clip = self.compute_dr_raster(vertices, faces)
        mask = self._compute_mask(rast, vertices_clip, faces)

        return mask, rast

    def render_position(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # Vertices in model coordinate system, real position coordinate number.
        faces = faces.to(torch.int32)
        mask, rast = self.render_rast_alpha(vertices, faces)

        vertices_model = vertices[None, ...].contiguous().float()
        position_map, _ = dr.interpolate(vertices_model, rast, faces)
        # Align with blender.
        if self.align_coordinate:
            position_map = position_map[..., [0, 2, 1]]
            position_map[..., 1] = -position_map[..., 1]

        position_map = torch.lerp(
            torch.zeros_like(position_map), position_map, mask
        )

        return position_map, mask

    def render_uv(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vtx_uv: torch.Tensor,
    ) -> Union[torch.Tensor, torch.Tensor]:
        faces = faces.to(torch.int32)
        mask, rast = self.render_rast_alpha(vertices, faces)
        uv_map, _ = dr.interpolate(vtx_uv, rast, faces)
        uv_map = torch.lerp(torch.zeros_like(uv_map), uv_map, mask)

        return uv_map, mask

    def render_depth(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # Vertices in model coordinate system, real depth coordinate number.
        faces = faces.to(torch.int32)
        mask, rast = self.render_rast_alpha(vertices, faces)

        vertices_camera = self.transform_vertices(vertices, matrix=self.mv_mtx)
        vertices_camera = vertices_camera[..., 2:3].contiguous().float()
        depth_map, _ = dr.interpolate(vertices_camera, rast, faces)
        # Change camera depth minus to positive.
        if self.align_coordinate:
            depth_map = -depth_map
        depth_map = torch.lerp(torch.zeros_like(depth_map), depth_map, mask)

        return depth_map, mask

    def render_global_normal(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertice_normals: torch.Tensor,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # NOTE: vertice_normals in [-1, 1],  return normal in [0, 1].
        # vertices / vertice_normals in model coordinate system.
        faces = faces.to(torch.int32)
        mask, rast = self.render_rast_alpha(vertices, faces)
        im_base_normals, _ = dr.interpolate(
            vertice_normals[None, ...].float(), rast, faces
        )

        if im_base_normals is not None:
            faces = faces.to(torch.int64)
            vertices_cam = self.transform_vertices(
                vertices, matrix=self.mv_mtx
            )
            face_vertices_ndc = kal.ops.mesh.index_vertices_by_faces(
                vertices_cam[..., :3], faces
            )
            face_normal_sign = kal.ops.mesh.face_normals(face_vertices_ndc)[
                ..., 2
            ]
            for idx in range(len(im_base_normals)):
                face_idx = (rast[idx, ..., -1].long() - 1).contiguous()
                im_normal_sign = torch.sign(face_normal_sign[idx, face_idx])
                im_normal_sign[face_idx == -1] = 0
                im_base_normals[idx] *= im_normal_sign.unsqueeze(-1)

        normal = (im_base_normals + 1) / 2
        normal = normal.clip(min=0, max=1)
        normal = torch.lerp(torch.zeros_like(normal), normal, mask)

        return normal, mask

    def transform_normal(
        self,
        normals: torch.Tensor,
        trans_matrix: torch.Tensor,
        masks: torch.Tensor,
        to_view: bool,
    ) -> torch.Tensor:
        # NOTE: input normals in [0, 1], output normals in [0, 1].
        normals = normals.clone()
        assert len(normals) == len(trans_matrix)

        if not to_view:
            # Flip the sign on the x-axis to match inv bae system for global transformation.  # noqa
            normals[..., 0] = 1 - normals[..., 0]

        normals = 2 * normals - 1
        b, h, w, c = normals.shape

        transformed_normals = []
        for normal, matrix in zip(normals, trans_matrix):
            # Transform normals using the transformation matrix (4x4).
            reshaped_normals = normal.view(-1, c)  # (h w 3) -> (hw 3)
            padded_vectors = torch.nn.functional.pad(
                reshaped_normals, pad=(0, 1), mode="constant", value=0.0
            )
            transformed_normal = torch.matmul(
                padded_vectors, matrix.transpose(0, 1)
            )[..., :3]

            # Normalize and clip the normals to [0, 1] range.
            transformed_normal = F.normalize(transformed_normal, p=2, dim=-1)
            transformed_normal = (transformed_normal + 1) / 2

            if to_view:
                # Flip the sign on the x-axis to match bae system for view transformation.  # noqa
                transformed_normal[..., 0] = 1 - transformed_normal[..., 0]

            transformed_normals.append(transformed_normal.view(h, w, c))

        transformed_normals = torch.stack(transformed_normals, dim=0)

        if masks is not None:
            transformed_normals = torch.lerp(
                torch.zeros_like(transformed_normals),
                transformed_normals,
                masks,
            )

        return transformed_normals


def _az_el_to_points(
    azimuths: np.ndarray, elevations: np.ndarray
) -> np.ndarray:
    x = np.cos(azimuths) * np.cos(elevations)
    y = np.sin(azimuths) * np.cos(elevations)
    z = np.sin(elevations)

    return np.stack([x, y, z], axis=-1)


def _compute_az_el_by_views(
    num_view: int, el: float
) -> Tuple[np.ndarray, np.ndarray]:
    azimuths = np.arange(num_view) / num_view * np.pi * 2
    elevations = np.deg2rad(np.array([el] * num_view))

    return azimuths, elevations


def _compute_cam_pts_by_az_el(
    azs: np.ndarray,
    els: np.ndarray,
    distance: float,
    extra_pts: np.ndarray = None,
) -> np.ndarray:
    distances = np.array([distance for _ in range(len(azs))])
    cam_pts = _az_el_to_points(azs, els) * distances[:, None]

    if extra_pts is not None:
        cam_pts = np.concatenate([cam_pts, extra_pts], axis=0)

    # Align coordinate system.
    cam_pts = cam_pts[:, [0, 2, 1]]  # xyz -> xzy
    cam_pts[..., 2] = -cam_pts[..., 2]

    return cam_pts


def compute_cam_pts_by_views(
    num_view: int, el: float, distance: float, extra_pts: np.ndarray = None
) -> torch.Tensor:
    """Computes object-center camera points for a given number of views.

    Args:
        num_view (int): The number of views (camera positions) to compute.
        el (float): The elevation angle in degrees.
        distance (float): The distance from the origin to the camera.
        extra_pts (np.ndarray): Extra camera points postion.

    Returns:
        torch.Tensor: A tensor containing the camera points for each view, with shape `(num_view, 3)`. # noqa
    """
    azimuths, elevations = _compute_az_el_by_views(num_view, el)
    cam_pts = _compute_cam_pts_by_az_el(
        azimuths, elevations, distance, extra_pts
    )

    return cam_pts


def save_images(
    images: Union[list[np.ndarray], list[torch.Tensor]],
    output_dir: str,
    cvt_color: str = None,
    format: str = ".png",
    to_uint8: bool = True,
    verbose: bool = False,
) -> List[str]:
    # NOTE: images in [0, 1]
    os.makedirs(output_dir, exist_ok=True)
    save_paths = []
    for idx, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if to_uint8:
            image = image.clip(min=0, max=1)
            image = (255.0 * image).astype(np.uint8)
        if cvt_color is not None:
            image = cv2.cvtColor(image, cvt_color)
        save_path = os.path.join(output_dir, f"{idx:04d}{format}")
        save_paths.append(save_path)

        cv2.imwrite(save_path, image)

    if verbose:
        logger.info(f"Images saved in {output_dir}")

    return save_paths


def _current_lighting(
    azimuths: List[float],
    elevations: List[float],
    light_factor: float = 1.0,
    device: str = "cuda",
):
    # azimuths, elevations in degress.
    directions = []
    for az, el in zip(azimuths, elevations):
        az, el = math.radians(az), math.radians(el)
        direction = kal.render.lighting.sg_direction_from_azimuth_elevation(
            az, el
        )
        directions.append(direction)
    directions = torch.cat(directions, dim=0)

    amplitude = torch.ones_like(directions) * light_factor
    light_condition = kal.render.lighting.SgLightingParameters(
        amplitude=amplitude,
        direction=directions,
        sharpness=3,
    ).to(device)

    # light_condition = kal.render.lighting.SgLightingParameters.from_sun(
    #     directions, strength=1, angle=90, color=None
    # ).to(device)

    return light_condition


def render_pbr(
    mesh,
    camera,
    device="cuda",
    cxt=None,
    custom_materials=None,
    light_factor=1.0,
):
    if cxt is None:
        cxt = dr.RasterizeCudaContext()

    light_condition = _current_lighting(
        azimuths=[0, 90, 180, 270],
        elevations=[90, 60, 30, 20],
        light_factor=light_factor,
        device=device,
    )
    render_res = kal.render.easy_render.render_mesh(
        camera,
        mesh,
        lighting=light_condition,
        nvdiffrast_context=cxt,
        custom_materials=custom_materials,
    )

    image = render_res[kal.render.easy_render.RenderPass.render]
    image = image.clip(0, 1)

    albedo = render_res[kal.render.easy_render.RenderPass.albedo]
    albedo = albedo.clip(0, 1)

    diffuse = render_res[kal.render.easy_render.RenderPass.diffuse]
    diffuse = diffuse.clip(0, 1)

    normal = render_res[kal.render.easy_render.RenderPass.normals]
    normal = normal.clip(-1, 1)

    return image, albedo, diffuse, normal


def _move_to_target_device(data, device: str):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = _move_to_target_device(value, device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)

    return data


def _encode_prompt(
    prompt_batch,
    text_encoders,
    tokenizers,
    proportion_empty_prompts=0,
    is_train=True,
):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(text_encoder.device)

            output = text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                position_ids=text_inputs.position_ids,
                output_hidden_states=True,
            )

            # We are only interested in the pooled output of the text encoder.
            prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
            pooled_prompt_embeds = output.hidden_states[-1][-1, :, :].clone()
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

    return prompt_embeds, pooled_prompt_embeds


def load_llm_models(pretrained_model_name_or_path: str, device: str):
    tokenizer = ChatGLMTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder = ChatGLMModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    ).to(device)

    text_encoders = [
        text_encoder,
    ]
    tokenizers = [
        tokenizer,
    ]

    logger.info(f"Load model from {pretrained_model_name_or_path} done.")

    return tokenizers, text_encoders


def prelabel_text_feature(
    prompt_batch: List[str],
    output_dir: str,
    tokenizers: nn.Module,
    text_encoders: nn.Module,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    # prompt_batch ["text..."]
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(
        prompt_batch, text_encoders, tokenizers
    )

    prompt_embeds = _move_to_target_device(prompt_embeds, device="cpu")
    pooled_prompt_embeds = _move_to_target_device(
        pooled_prompt_embeds, device="cpu"
    )

    data_dict = dict(
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds
    )

    save_path = os.path.join(output_dir, "text_feat.pth")
    torch.save(data_dict, save_path)

    return save_path


def _calc_face_normals(
    vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
    faces: torch.Tensor,  # F,3 long, first face may be all zero
    normalize: bool = False,
) -> torch.Tensor:  # F,3
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # F,3
    if normalize:
        face_normals = F.normalize(
            face_normals, eps=1e-6, dim=1
        )  # TODO inplace?
    return face_normals  # F,3


def calc_vertex_normals(
    vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
    faces: torch.Tensor,  # F,3 long, first face may be all zero
    face_normals: torch.Tensor = None,  # F,3, not normalized
) -> torch.Tensor:  # F,3
    _F = faces.shape[0]

    if face_normals is None:
        face_normals = _calc_face_normals(vertices, faces)

    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device
    )  # V,C=3,3
    vertex_normals.scatter_add_(
        dim=0,
        index=faces[:, :, None].expand(_F, 3, 3),
        src=face_normals[:, None, :].expand(_F, 3, 3),
    )
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return F.normalize(vertex_normals, eps=1e-6, dim=1)


def normalize_vertices_array(
    vertices: Union[torch.Tensor, np.ndarray],
    mesh_scale: float = 1.0,
    exec_norm: bool = True,
):
    if isinstance(vertices, torch.Tensor):
        bbmin, bbmax = vertices.min(0)[0], vertices.max(0)[0]
    else:
        bbmin, bbmax = vertices.min(0), vertices.max(0)  # (3,)
    center = (bbmin + bbmax) * 0.5
    bbsize = bbmax - bbmin
    scale = 2 * mesh_scale / bbsize.max()
    if exec_norm:
        vertices = (vertices - center) * scale

    return vertices, scale, center


def load_mesh_to_unit_cube(
    mesh_file: str,
    mesh_scale: float = 1.0,
) -> tuple[trimesh.Trimesh, float, list[float]]:
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"mesh_file path {mesh_file} not exists.")

    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.utils.concatenate(mesh)

    vertices, scale, center = normalize_vertices_array(
        mesh.vertices, mesh_scale
    )
    mesh.vertices = vertices

    return mesh, scale, center


def as_list(obj):
    if isinstance(obj, (list, tuple)):
        return obj
    elif isinstance(obj, set):
        return list(obj)
    else:
        return [obj]


@dataclass
class CameraSetting:
    """Camera settings for images rendering."""

    num_images: int
    elevation: list[float]
    distance: float
    resolution_hw: tuple[int, int]
    fov: float
    at: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )
    up: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 1.0, 0.0)
    )
    device: str = "cuda"
    near: float = 1e-2
    far: float = 1e2

    def __post_init__(
        self,
    ):
        h = self.resolution_hw[0]
        f = (h / 2) / math.tan(self.fov / 2)
        cx = self.resolution_hw[1] / 2
        cy = self.resolution_hw[0] / 2
        Ks = [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ]

        self.Ks = Ks


def _compute_az_el_by_camera_params(
    camera_params: CameraSetting, flip_az: bool = False
):
    num_view = camera_params.num_images // len(camera_params.elevation)
    view_interval = 2 * np.pi / num_view / 2
    azimuths = []
    elevations = []
    for idx, el in enumerate(camera_params.elevation):
        azs = np.arange(num_view) / num_view * np.pi * 2 + idx * view_interval
        if flip_az:
            azs *= -1
        els = np.deg2rad(np.array([el] * num_view))
        azimuths.append(azs)
        elevations.append(els)

    azimuths = np.concatenate(azimuths, axis=0)
    elevations = np.concatenate(elevations, axis=0)

    return azimuths, elevations


def init_kal_camera(camera_params: CameraSetting) -> Camera:
    azimuths, elevations = _compute_az_el_by_camera_params(camera_params)
    cam_pts = _compute_cam_pts_by_az_el(
        azimuths, elevations, camera_params.distance
    )

    up = torch.cat(
        [
            torch.tensor(camera_params.up).repeat(camera_params.num_images, 1),
        ],
        dim=0,
    )

    camera = Camera.from_args(
        eye=torch.tensor(cam_pts),
        at=torch.tensor(camera_params.at),
        up=up,
        fov=camera_params.fov,
        height=camera_params.resolution_hw[0],
        width=camera_params.resolution_hw[1],
        near=camera_params.near,
        far=camera_params.far,
        device=camera_params.device,
    )

    return camera


def import_kaolin_mesh(mesh_path: str, with_mtl: bool = False):
    if mesh_path.endswith(".glb"):
        mesh = kal.io.gltf.import_mesh(mesh_path)
    elif mesh_path.endswith(".obj"):
        with_material = True if with_mtl else False
        mesh = kal.io.obj.import_mesh(mesh_path, with_materials=with_material)
        if with_mtl and mesh.materials and len(mesh.materials) > 0:
            material = kal.render.materials.PBRMaterial()
            assert (
                "map_Kd" in mesh.materials[0]
            ), "'map_Kd' not found in materials."
            material.diffuse_texture = mesh.materials[0]["map_Kd"] / 255.0
            mesh.materials = [material]
    elif mesh_path.endswith(".ply"):
        mesh = trimesh.load(mesh_path)
        mesh_path = mesh_path.replace(".ply", ".obj")
        mesh.export(mesh_path)
        mesh = kal.io.obj.import_mesh(mesh_path)
    elif mesh_path.endswith(".off"):
        mesh = kal.io.off.import_mesh(mesh_path)
    else:
        raise RuntimeError(
            f"{mesh_path} mesh type not supported, "
            "supported mesh type `.glb`, `.obj`, `.ply`, `.off`."
        )

    return mesh


def save_mesh_with_mtl(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture: Union[Image.Image, np.ndarray],
    output_path: str,
    material_base=(250, 250, 250, 255),
) -> trimesh.Trimesh:
    if isinstance(texture, np.ndarray):
        texture = Image.fromarray(texture)

    mesh = trimesh.Trimesh(
        vertices,
        faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, image=texture),
    )
    mesh.visual.material = trimesh.visual.material.SimpleMaterial(
        image=texture,
        diffuse=material_base,
        ambient=material_base,
        specular=material_base,
    )

    dir_name = os.path.dirname(output_path)
    os.makedirs(dir_name, exist_ok=True)

    _ = mesh.export(output_path)
    # texture.save(os.path.join(dir_name, f"{file_name}_texture.png"))

    logger.info(f"Saved mesh with texture to {output_path}")

    return mesh


def get_images_from_grid(
    image: Union[str, Image.Image], img_size: int
) -> list[Image.Image]:
    if isinstance(image, str):
        image = Image.open(image)

    view_images = np.array(image)
    view_images = np.concatenate(
        [view_images[:img_size, ...], view_images[img_size:, ...]], axis=1
    )
    images = np.split(view_images, view_images.shape[1] // img_size, axis=1)
    images = [Image.fromarray(img) for img in images]

    return images


def post_process_texture(texture: np.ndarray, iter: int = 1) -> np.ndarray:
    for _ in range(iter):
        texture = cv2.fastNlMeansDenoisingColored(texture, None, 2, 2, 7, 15)
        texture = cv2.bilateralFilter(
            texture, d=5, sigmaColor=20, sigmaSpace=20
        )

    return texture


def quat_mult(q1, q2):
    # NOTE:
    # Q1 is the quaternion that rotates the vector from the original position to the final position  # noqa
    # Q2 is the quaternion that been rotated
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def quat_to_rotmat(quats: torch.Tensor, mode="wxyz") -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)

    if mode == "xyzw":
        x, y, z, w = torch.unbind(quats, dim=-1)
    elif mode == "wxyz":
        w, x, y, z = torch.unbind(quats, dim=-1)
    else:
        raise ValueError(f"Invalid mode: {mode}.")

    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )

    return R.reshape(quats.shape[:-1] + (3, 3))


def gamma_shs(shs: torch.Tensor, gamma: float) -> torch.Tensor:
    C0 = 0.28209479177387814  # Constant for normalization in spherical harmonics  # noqa
    # Clip to the range [0.0, 1.0], apply gamma correction, and then un-clip back  # noqa
    new_shs = torch.clip(shs * C0 + 0.5, 0.0, 1.0)
    new_shs = (torch.pow(new_shs, gamma) - 0.5) / C0
    return new_shs


def resize_pil(image: Image.Image, max_size: int = 1024) -> Image.Image:
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def trellis_preprocess(image: Image.Image) -> Image.Image:
    """Process the input image as trellis done."""
    image_np = np.array(image)
    alpha = image_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = (
        np.min(bbox[:, 1]),
        np.min(bbox[:, 0]),
        np.max(bbox[:, 1]),
        np.max(bbox[:, 0]),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    image = image.crop(bbox)
    image = image.resize((518, 518), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255
    image = image[:, :, :3] * image[:, :, 3:4]
    image = Image.fromarray((image * 255).astype(np.uint8))

    return image


def zip_files(input_paths: list[str], output_zip: str) -> str:
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for input_path in input_paths:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File not found: {input_path}")

            if os.path.isdir(input_path):
                for root, _, files in os.walk(input_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(
                            file_path, start=os.path.commonpath(input_paths)
                        )
                        zipf.write(file_path, arcname=arcname)
            else:
                arcname = os.path.relpath(
                    input_path, start=os.path.commonpath(input_paths)
                )
                zipf.write(input_path, arcname=arcname)

    return output_zip


def delete_dir(folder_path: str, keep_subs: list[str] = None) -> None:
    for item in os.listdir(folder_path):
        if keep_subs is not None and item in keep_subs:
            continue
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            rmtree(item_path)
        else:
            os.remove(item_path)
