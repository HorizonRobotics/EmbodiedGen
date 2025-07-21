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
# Part of the code comes from https://github.com/nerfstudio-project/gsplat
# Both under the Apache License, Version 2.0.


import math
import random
from io import BytesIO
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import trimesh
from gsplat.optimizers import SelectiveAdam
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from embodied_gen.models.gs_model import GaussianOperator

__all__ = [
    "set_random_seed",
    "export_splats",
    "create_splats_with_optimizers",
    "compute_pinhole_intrinsics",
    "resize_pinhole_intrinsics",
    "restore_scene_scale_and_position",
]


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def splat2ply_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
) -> bytes:
    num_splats = means.shape[0]
    buffer = BytesIO()

    # Write PLY header
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element vertex {num_splats}\n".encode())
    buffer.write(b"property float x\n")
    buffer.write(b"property float y\n")
    buffer.write(b"property float z\n")
    for i, data in enumerate([sh0, shN]):
        prefix = "f_dc" if i == 0 else "f_rest"
        for j in range(data.shape[1]):
            buffer.write(f"property float {prefix}_{j}\n".encode())
    buffer.write(b"property float opacity\n")
    for i in range(scales.shape[1]):
        buffer.write(f"property float scale_{i}\n".encode())
    for i in range(quats.shape[1]):
        buffer.write(f"property float rot_{i}\n".encode())
    buffer.write(b"end_header\n")

    # Concatenate all tensors in the correct order
    splat_data = torch.cat(
        [means, sh0, shN, opacities.unsqueeze(1), scales, quats], dim=1
    )
    # Ensure correct dtype
    splat_data = splat_data.to(torch.float32)

    # Write binary data
    float_dtype = np.dtype(np.float32).newbyteorder("<")
    buffer.write(
        splat_data.detach().cpu().numpy().astype(float_dtype).tobytes()
    )

    return buffer.getvalue()


def export_splats(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    format: Literal["ply"] = "ply",
    save_to: Optional[str] = None,
) -> bytes:
    """Export a Gaussian Splats model to bytes in PLY file format."""
    total_splats = means.shape[0]
    assert means.shape == (total_splats, 3), "Means must be of shape (N, 3)"
    assert scales.shape == (total_splats, 3), "Scales must be of shape (N, 3)"
    assert quats.shape == (
        total_splats,
        4,
    ), "Quaternions must be of shape (N, 4)"
    assert opacities.shape == (
        total_splats,
    ), "Opacities must be of shape (N,)"
    assert sh0.shape == (total_splats, 1, 3), "sh0 must be of shape (N, 1, 3)"
    assert (
        shN.ndim == 3 and shN.shape[0] == total_splats and shN.shape[2] == 3
    ), f"shN must be of shape (N, K, 3), got {shN.shape}"

    # Reshape spherical harmonics
    sh0 = sh0.squeeze(1)  # Shape (N, 3)
    shN = shN.permute(0, 2, 1).reshape(means.shape[0], -1)  # Shape (N, K * 3)

    # Check for NaN or Inf values
    invalid_mask = (
        torch.isnan(means).any(dim=1)
        | torch.isinf(means).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isinf(scales).any(dim=1)
        | torch.isnan(quats).any(dim=1)
        | torch.isinf(quats).any(dim=1)
        | torch.isnan(opacities).any(dim=0)
        | torch.isinf(opacities).any(dim=0)
        | torch.isnan(sh0).any(dim=1)
        | torch.isinf(sh0).any(dim=1)
        | torch.isnan(shN).any(dim=1)
        | torch.isinf(shN).any(dim=1)
    )

    # Filter out invalid entries
    valid_mask = ~invalid_mask
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    opacities = opacities[valid_mask]
    sh0 = sh0[valid_mask]
    shN = shN[valid_mask]

    if format == "ply":
        data = splat2ply_bytes(means, scales, quats, opacities, sh0, shN)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if save_to:
        with open(save_to, "wb") as binary_file:
            binary_file.write(data)

    return data


def create_splats_with_optimizers(
    points: np.ndarray = None,
    points_rgb: np.ndarray = None,
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if points is not None and points_rgb is not None:
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(points_rgb / 255.0).float()
    else:
        points = (
            init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        )
        rgbs = torch.rand((init_num_pts, 3))

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = (
        torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
    )  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def compute_pinhole_intrinsics(
    image_w: int, image_h: int, fov_deg: float
) -> np.ndarray:
    fov_rad = np.deg2rad(fov_deg)
    fx = image_w / (2 * np.tan(fov_rad / 2))
    fy = fx  # assuming square pixels
    cx = image_w / 2
    cy = image_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


def resize_pinhole_intrinsics(
    raw_K: np.ndarray | torch.Tensor,
    raw_hw: tuple[int, int],
    new_hw: tuple[int, int],
) -> np.ndarray:
    raw_h, raw_w = raw_hw
    new_h, new_w = new_hw

    scale_x = new_w / raw_w
    scale_y = new_h / raw_h

    new_K = raw_K.copy() if isinstance(raw_K, np.ndarray) else raw_K.clone()
    new_K[0, 0] *= scale_x  # fx
    new_K[0, 2] *= scale_x  # cx
    new_K[1, 1] *= scale_y  # fy
    new_K[1, 2] *= scale_y  # cy

    return new_K


def restore_scene_scale_and_position(
    real_height: float, mesh_path: str, gs_path: str
) -> None:
    """Scales a mesh and corresponding GS model to match a given real-world height.

    Uses the 1st and 99th percentile of mesh Z-axis to estimate height,
    applies scaling and vertical alignment, and updates both the mesh and GS model.

    Args:
        real_height (float): Target real-world height among Z axis.
        mesh_path (str): Path to the input mesh file.
        gs_path (str): Path to the Gaussian Splatting model file.
    """
    mesh = trimesh.load(mesh_path)
    z_min = np.percentile(mesh.vertices[:, 1], 1)
    z_max = np.percentile(mesh.vertices[:, 1], 99)
    height = z_max - z_min
    scale = real_height / height

    rot = Rotation.from_quat([0, 1, 0, 0])
    mesh.vertices = rot.apply(mesh.vertices)
    mesh.vertices[:, 1] -= z_min
    mesh.vertices *= scale
    mesh.export(mesh_path)

    gs_model: GaussianOperator = GaussianOperator.load_from_ply(gs_path)
    gs_model = gs_model.get_gaussians(
        instance_pose=torch.tensor([0.0, -z_min, 0, 0, 1, 0, 0])
    )
    gs_model.rescale(scale)
    gs_model.save_to_ply(gs_path)
