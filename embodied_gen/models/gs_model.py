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
import os
import struct
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.rendering import rasterization
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from embodied_gen.data.utils import gamma_shs, quat_mult, quat_to_rotmat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "RenderResult",
    "GaussianOperator",
]


@dataclass
class RenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    opacity: np.ndarray
    mask_threshold: float = 10
    mask: Optional[np.ndarray] = None
    rgba: Optional[np.ndarray] = None

    def __post_init__(self):
        if isinstance(self.rgb, torch.Tensor):
            rgb = (self.rgb * 255).to(torch.uint8)
            self.rgb = rgb.cpu().numpy()[..., ::-1]
        if isinstance(self.depth, torch.Tensor):
            self.depth = self.depth.cpu().numpy()
        if isinstance(self.opacity, torch.Tensor):
            opacity = (self.opacity * 255).to(torch.uint8)
            self.opacity = opacity.cpu().numpy()
            mask = np.where(self.opacity > self.mask_threshold, 255, 0)
            self.mask = mask.astype(np.uint8)
            self.rgba = np.concatenate([self.rgb, self.mask], axis=-1)


@dataclass
class GaussianBase:
    _opacities: torch.Tensor
    _means: torch.Tensor
    _scales: torch.Tensor
    _quats: torch.Tensor
    _rgbs: Optional[torch.Tensor] = None
    _features_dc: Optional[torch.Tensor] = None
    _features_rest: Optional[torch.Tensor] = None
    sh_degree: Optional[int] = 0
    device: str = "cuda"

    def __post_init__(self):
        self.active_sh_degree: int = self.sh_degree
        self.to(self.device)

    def to(self, device: str) -> None:
        for k, v in self.__dict__.items():
            if not isinstance(v, torch.Tensor):
                continue
            self.__dict__[k] = v.to(device)

    def get_numpy_data(self):
        data = {}
        for k, v in self.__dict__.items():
            if not isinstance(v, torch.Tensor):
                continue
            data[k] = v.detach().cpu().numpy()

        return data

    def quat_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    @classmethod
    def load_from_ply(
        cls,
        path: str,
        gamma: float = 1.0,
        device: str = "cuda",
    ) -> "GaussianBase":
        plydata = PlyData.read(path)
        xyz = torch.stack(
            (
                torch.tensor(plydata.elements[0]["x"], dtype=torch.float32),
                torch.tensor(plydata.elements[0]["y"], dtype=torch.float32),
                torch.tensor(plydata.elements[0]["z"], dtype=torch.float32),
            ),
            dim=1,
        )

        opacities = torch.tensor(
            plydata.elements[0]["opacity"], dtype=torch.float32
        ).unsqueeze(-1)
        features_dc = torch.zeros((xyz.shape[0], 3), dtype=torch.float32)
        features_dc[:, 0] = torch.tensor(
            plydata.elements[0]["f_dc_0"], dtype=torch.float32
        )
        features_dc[:, 1] = torch.tensor(
            plydata.elements[0]["f_dc_1"], dtype=torch.float32
        )
        features_dc[:, 2] = torch.tensor(
            plydata.elements[0]["f_dc_2"], dtype=torch.float32
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = torch.zeros(
            (xyz.shape[0], len(scale_names)), dtype=torch.float32
        )
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = torch.tensor(
                plydata.elements[0][attr_name], dtype=torch.float32
            )

        rot_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("rot_")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = torch.zeros((xyz.shape[0], len(rot_names)), dtype=torch.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = torch.tensor(
                plydata.elements[0][attr_name], dtype=torch.float32
            )

        rots = rots / torch.norm(rots, dim=-1, keepdim=True)

        # extra features
        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split("_")[-1])
        )

        max_sh_degree = int(np.sqrt((len(extra_f_names) + 3) / 3) - 1)
        if max_sh_degree != 0:
            features_extra = torch.zeros(
                (xyz.shape[0], len(extra_f_names)), dtype=torch.float32
            )
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = torch.tensor(
                    plydata.elements[0][attr_name], dtype=torch.float32
                )

            features_extra = features_extra.view(
                (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
            )
            features_extra = features_extra.permute(0, 2, 1)

            if abs(gamma - 1.0) > 1e-3:
                features_dc = gamma_shs(features_dc, gamma)
                features_extra[..., :] = 0.0
                opacities *= 0.8

            shs = torch.cat(
                [
                    features_dc.reshape(-1, 3),
                    features_extra.reshape(len(features_dc), -1),
                ],
                dim=-1,
            )
        else:
            # sh_dim is 0, only dc features
            shs = features_dc
            features_extra = None

        return cls(
            sh_degree=max_sh_degree,
            _means=xyz,
            _opacities=opacities,
            _rgbs=shs,
            _scales=scales,
            _quats=rots,
            _features_dc=features_dc,
            _features_rest=features_extra,
            device=device,
        )

    def save_to_ply(
        self, path: str, colors: torch.Tensor = None, enable_mask: bool = False
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        numpy_data = self.get_numpy_data()
        means = numpy_data["_means"]
        scales = numpy_data["_scales"]
        quats = numpy_data["_quats"]
        opacities = numpy_data["_opacities"]
        sh0 = numpy_data["_features_dc"]
        shN = numpy_data.get("_features_rest", np.zeros((means.shape[0], 0)))
        shN = shN.reshape(means.shape[0], -1)

        # Create a mask to identify rows with NaN or Inf in any of the numpy_data arrays  # noqa
        if enable_mask:
            invalid_mask = (
                np.isnan(means).any(axis=1)
                | np.isinf(means).any(axis=1)
                | np.isnan(scales).any(axis=1)
                | np.isinf(scales).any(axis=1)
                | np.isnan(quats).any(axis=1)
                | np.isinf(quats).any(axis=1)
                | np.isnan(opacities).any(axis=0)
                | np.isinf(opacities).any(axis=0)
                | np.isnan(sh0).any(axis=1)
                | np.isinf(sh0).any(axis=1)
                | np.isnan(shN).any(axis=1)
                | np.isinf(shN).any(axis=1)
            )

            # Filter out rows with NaNs or Infs from all data arrays
            means = means[~invalid_mask]
            scales = scales[~invalid_mask]
            quats = quats[~invalid_mask]
            opacities = opacities[~invalid_mask]
            sh0 = sh0[~invalid_mask]
            shN = shN[~invalid_mask]

        num_points = means.shape[0]

        with open(path, "wb") as f:
            # Write PLY header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {num_points}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")

            if colors is not None:
                for j in range(colors.shape[1]):
                    f.write(f"property float f_dc_{j}\n".encode())
            else:
                for i, data in enumerate([sh0, shN]):
                    prefix = "f_dc" if i == 0 else "f_rest"
                    for j in range(data.shape[1]):
                        f.write(f"property float {prefix}_{j}\n".encode())

            f.write(b"property float opacity\n")

            for i in range(scales.shape[1]):
                f.write(f"property float scale_{i}\n".encode())
            for i in range(quats.shape[1]):
                f.write(f"property float rot_{i}\n".encode())

            f.write(b"end_header\n")

            # Write vertex data
            for i in range(num_points):
                f.write(struct.pack("<fff", *means[i]))  # x, y, z
                f.write(struct.pack("<fff", 0, 0, 0))  # nx, ny, nz (zeros)

                if colors is not None:
                    color = colors.detach().cpu().numpy()
                    for j in range(color.shape[1]):
                        f_dc = (color[i, j] - 0.5) / 0.2820947917738781
                        f.write(struct.pack("<f", f_dc))
                else:
                    for data in [sh0, shN]:
                        for j in range(data.shape[1]):
                            f.write(struct.pack("<f", data[i, j]))

                f.write(struct.pack("<f", opacities[i]))  # opacity

                for data in [scales, quats]:
                    for j in range(data.shape[1]):
                        f.write(struct.pack("<f", data[i, j]))


@dataclass
class GaussianOperator(GaussianBase):
    """Gaussian Splatting operator.

    Supports transformation, scaling, color computation, and
    rasterization-based rendering.

    Inherits:
        GaussianBase: Base class with Gaussian params (means, scales, etc.)

    Functionality includes:
    - Applying instance poses to transform Gaussian means and quaternions.
    - Scaling Gaussians to a real-world size.
    - Computing colors using spherical harmonics.
    - Rendering images via differentiable rasterization.
    - Exporting transformed and rescaled models to .ply format.
    """

    def _compute_transform(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        instance_pose: torch.Tensor,
    ):
        """Compute the transform of the GS models.

        Args:
            means: tensor of gs means.
            quats: tensor of gs quaternions.
            instance_pose: instances poses in [x y z qx qy qz qw] format.

        """
        # (x y z qx qy qz qw) -> (x y z qw qx qy qz)
        instance_pose = instance_pose[[0, 1, 2, 6, 3, 4, 5]]
        cur_instances_quats = self.quat_norm(instance_pose[3:])
        rot_cur = quat_to_rotmat(cur_instances_quats, mode="wxyz")

        # update the means
        num_gs = means.shape[0]
        trans_per_pts = torch.stack([instance_pose[:3]] * num_gs, dim=0)
        quat_per_pts = torch.stack([instance_pose[3:]] * num_gs, dim=0)
        rot_per_pts = torch.stack([rot_cur] * num_gs, dim=0)  # (num_gs, 3, 3)

        # update the means
        cur_means = (
            torch.bmm(rot_per_pts, means.unsqueeze(-1)).squeeze(-1)
            + trans_per_pts
        )

        # update the quats
        _quats = self.quat_norm(quats)
        cur_quats = quat_mult(quat_per_pts, _quats)

        return cur_means, cur_quats

    def get_gaussians(
        self,
        c2w: torch.Tensor = None,
        instance_pose: torch.Tensor = None,
        apply_activate: bool = False,
    ) -> "GaussianBase":
        """Get Gaussian data under the given instance_pose."""
        if c2w is None:
            c2w = torch.eye(4).to(self.device)

        if instance_pose is not None:
            # compute the transformed gs means and quats
            world_means, world_quats = self._compute_transform(
                self._means, self._quats, instance_pose.float().to(self.device)
            )
        else:
            world_means, world_quats = self._means, self._quats

        # get colors of gaussians
        if self._features_rest is not None:
            colors = torch.cat(
                (self._features_dc[:, None, :], self._features_rest), dim=1
            )
        else:
            colors = self._features_dc[:, None, :]

        if self.sh_degree > 0:
            viewdirs = world_means.detach() - c2w[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        gs_dict = dict(
            _means=world_means,
            _opacities=(
                torch.sigmoid(self._opacities)
                if apply_activate
                else self._opacities
            ),
            _rgbs=rgbs,
            _scales=(
                torch.exp(self._scales) if apply_activate else self._scales
            ),
            _quats=self.quat_norm(world_quats),
            _features_dc=self._features_dc,
            _features_rest=self._features_rest,
            sh_degree=self.sh_degree,
            device=self.device,
        )

        return GaussianOperator(**gs_dict)

    def rescale(self, scale: float):
        if scale != 1.0:
            self._means *= scale
            self._scales += torch.log(self._scales.new_tensor(scale))

    def set_scale_by_height(self, real_height: float) -> None:
        def _ptp(tensor, dim):
            val = tensor.max(dim=dim).values - tensor.min(dim=dim).values
            return val.tolist()

        xyz_scale = max(_ptp(self._means, dim=0))
        self.rescale(1 / (xyz_scale + 1e-6))  # Normalize to [-0.5, 0.5]
        raw_height = _ptp(self._means, dim=0)[1]
        scale = real_height / raw_height

        self.rescale(scale)

        return

    @staticmethod
    def resave_ply(
        in_ply: str,
        out_ply: str,
        real_height: float = None,
        instance_pose: np.ndarray = None,
        device: str = "cuda",
    ) -> None:
        gs_model = GaussianOperator.load_from_ply(in_ply, device=device)

        if instance_pose is not None:
            gs_model = gs_model.get_gaussians(instance_pose=instance_pose)

        if real_height is not None:
            gs_model.set_scale_by_height(real_height)

        gs_model.save_to_ply(out_ply)

        return

    @staticmethod
    def trans_to_quatpose(
        rot_matrix: list[list[float]],
        trans_matrix: list[float] = [0, 0, 0],
    ) -> torch.Tensor:
        if isinstance(rot_matrix, list):
            rot_matrix = np.array(rot_matrix)

        rot = Rotation.from_matrix(rot_matrix)
        qx, qy, qz, qw = rot.as_quat()
        instance_pose = torch.tensor([*trans_matrix, qx, qy, qz, qw])

        return instance_pose

    def render(
        self,
        c2w: torch.Tensor,
        Ks: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderResult:
        gs = self.get_gaussians(c2w, apply_activate=True)
        renders, alphas, _ = rasterization(
            means=gs._means,
            quats=gs._quats,
            scales=gs._scales,
            opacities=gs._opacities.squeeze(),
            colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None, ...],
            Ks=Ks[None, ...],
            width=image_width,
            height=image_height,
            packed=False,
            absgrad=True,
            sparse_grad=False,
            # rasterize_mode="classic",
            rasterize_mode="antialiased",
            **{
                "near_plane": 0.01,
                "far_plane": 1000000000,
                "radius_clip": 0.0,
                "render_mode": "RGB+ED",
            },
        )
        renders = renders[0]
        alphas = alphas[0].squeeze(-1)

        assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
        rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)

        return RenderResult(
            torch.clamp(rendered_rgb, min=0, max=1),
            rendered_depth,
            alphas[..., None],
        )


if __name__ == "__main__":
    input_gs = "outputs/test/debug.ply"
    output_gs = "./debug_v3.ply"
    gs_model: GaussianOperator = GaussianOperator.load_from_ply(input_gs)

    # 绕 x 轴旋转 180°
    R_x = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    instance_pose = gs_model.trans_to_quatpose(R_x)
    gs_model = gs_model.get_gaussians(instance_pose=instance_pose)

    gs_model.rescale(2)

    gs_model.set_scale_by_height(1.3)

    gs_model.save_to_ply(output_gs)
