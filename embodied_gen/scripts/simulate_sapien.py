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


import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import imageio
import numpy as np
import torch
import tyro
from tqdm import tqdm
from embodied_gen.models.gs_model import GaussianOperator
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import alpha_blend_rgba
from embodied_gen.utils.simulation import SIM_COORD_ALIGN, SapienSceneManager


@dataclass
class SapienSimConfig:
    # Simulation settings.
    layout_path: str
    output_dir: str
    sim_freq: int = 250
    sim_step: int = 600
    z_offset: float = 0.02
    init_quat: list[float] = field(
        default_factory=lambda: [0.7071, 0, 0, 0.7071]
    )  # xyzw
    device: str = "cuda"
    # Camera settings.
    render_interval: int = 8
    num_cameras: int = 3
    camera_radius: float = 0.9
    camera_height: float = 1.1
    image_hw: tuple[int, int] = (768, 1024)
    fovy_deg: float = 75.0
    camera_target_pt: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.9]
    )
    render_keys: list[
        Literal[
            "Color", "Foreground", "Segmentation", "Normal", "Mask", "Depth"
        ]
    ] = field(default_factory=lambda: ["Foreground"])


def entrypoint(**kwargs):
    if kwargs is None or len(kwargs) == 0:
        cfg = tyro.cli(SapienSimConfig)
    else:
        cfg = SapienSimConfig(**kwargs)

    scene_manager = SapienSceneManager(cfg.sim_freq)
    _ = scene_manager.initialize_circular_cameras(
        num_cameras=cfg.num_cameras,
        radius=cfg.camera_radius,
        height=cfg.camera_height,
        target_pt=cfg.camera_target_pt,
        image_hw=cfg.image_hw,
        fovy_deg=cfg.fovy_deg,
    )
    with open(cfg.layout_path, "r") as f:
        layout_data = json.load(f)
        layout_data: LayoutInfo = LayoutInfo.from_dict(layout_data)

    scene_manager.load_assets_from_layout_file(
        layout_data, cfg.z_offset, cfg.init_quat
    )

    frames = defaultdict(list)
    image_cnt = 0
    for step in tqdm(range(cfg.sim_step), desc="Simulation"):
        scene_manager.scene.step()
        if step % cfg.render_interval != 0:
            continue
        scene_manager.scene.update_render()
        image_cnt += 1
        for camera in scene_manager.cameras:
            camera.take_picture()
            images = scene_manager.render_images(
                camera, render_keys=cfg.render_keys
            )
            frames[camera.name].append(images)

    if "Foreground" in cfg.render_keys:
        bg_node = layout_data.relation[Scene3DItemEnum.BACKGROUND.value]
        gs_path = f"{layout_data.assets[bg_node]}/gs_model.ply"

        gs_model: GaussianOperator = GaussianOperator.load_from_ply(gs_path)
        init_pose = torch.tensor([0, 0, 0] + cfg.init_quat)
        gs_model = gs_model.get_gaussians(instance_pose=init_pose)

        bg_images = dict()
        for camera in scene_manager.cameras:
            Ks = camera.get_intrinsic_matrix()
            c2w = camera.get_model_matrix()
            c2w = c2w @ SIM_COORD_ALIGN
            result = gs_model.render(
                torch.tensor(c2w, dtype=torch.float32).to(cfg.device),
                torch.tensor(Ks, dtype=torch.float32).to(cfg.device),
                image_width=cfg.image_hw[1],
                image_height=cfg.image_hw[0],
            )
            bg_images[camera.name] = result.rgb[..., ::-1]

        video_frames = []
        for camera in scene_manager.cameras:
            for step in tqdm(range(image_cnt), desc="Rendering"):
                rgba = alpha_blend_rgba(
                    frames[camera.name][step]["Foreground"],
                    bg_images[camera.name],
                )
                video_frames.append(np.array(rgba))

        os.makedirs(cfg.output_dir, exist_ok=True)
        video_path = f"{cfg.output_dir}/Iscene.mp4"
        imageio.mimsave(video_path, video_frames, fps=20)
        logger.info(f"Interative 3D Scene Visualization saved in {video_path}")


if __name__ == "__main__":
    entrypoint()
