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

import os
import sys
from collections import defaultdict

import numpy as np
import spaces
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, "../.."))
from thirdparty.TRELLIS.trellis.renderers import GaussianRenderer, MeshRenderer
from thirdparty.TRELLIS.trellis.representations import (
    Gaussian,
    MeshExtractResult,
)
from thirdparty.TRELLIS.trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)

__all__ = [
    "render_video",
    "pack_state",
    "unpack_state",
]


@spaces.GPU
def render_mesh_frames(sample, extrinsics, intrinsics, options={}, **kwargs):
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
def render_gs_frames(
    sample,
    extrinsics,
    intrinsics,
    options=None,
    colors_overwrite=None,
    verbose=True,
    **kwargs,
):
    def to_img(tensor):
        return np.clip(
            tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
        ).astype(np.uint8)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    renderer = GaussianRenderer()
    renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)
    renderer.pipe.use_mip_gaussian = True

    defaults = {
        "resolution": 512,
        "near": 0.8,
        "far": 1.6,
        "bg_color": (0, 0, 0),
        "ssaa": 1,
    }
    final_options = {**defaults, **(options or {})}

    for k, v in final_options.items():
        if hasattr(renderer.rendering_options, k):
            setattr(renderer.rendering_options, k, v)

    outputs = defaultdict(list)
    iterator = zip(extrinsics, intrinsics)
    if verbose:
        iterator = tqdm(iterator, total=len(extrinsics), desc="Rendering")

    for extr, intr in iterator:
        res = renderer.render(
            sample, extr, intr, colors_overwrite=colors_overwrite
        )
        outputs["color"].append(to_img(res["color"]))
        depth = res.get("percent_depth") or res.get("depth")
        outputs["depth"].append(to_numpy(depth) if depth is not None else None)

    return dict(outputs)


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
        render_mesh_frames
        if sample.__class__.__name__ == "MeshExtractResult"
        else render_gs_frames
    )
    result = render_fn(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )

    return result


@spaces.GPU
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict, device: str = "cpu") -> tuple[Gaussian, dict]:
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
        device=device,
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device=device)
    gs._features_dc = torch.tensor(
        state["gaussian"]["_features_dc"], device=device
    )
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device=device)
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device=device)
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device=device)

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device=device),
        faces=torch.tensor(state["mesh"]["faces"], device=device),
    )

    return gs, mesh
