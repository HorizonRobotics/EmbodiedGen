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
from embodied_gen.utils.geometry import quaternion_multiply
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import alpha_blend_rgba
from embodied_gen.utils.simulation import (
    SIM_COORD_ALIGN,
    FrankaPandaGrasper,
    SapienSceneManager,
    load_assets_from_layout_file,
    load_mani_skill_robot,
    render_images,
)


@dataclass
class SapienSimConfig:
    # Simulation settings.
    layout_path: str
    output_dir: str
    sim_freq: int = 200
    sim_step: int = 400
    z_offset: float = 0.004
    init_quat: list[float] = field(
        default_factory=lambda: [0.7071, 0, 0, 0.7071]
    )  # xyzw
    device: str = "cuda"
    control_freq: int = 50
    insert_robot: bool = False
    # Camera settings.
    render_interval: int = 10
    num_cameras: int = 3
    camera_radius: float = 0.9
    camera_height: float = 1.1
    image_hw: tuple[int, int] = (512, 512)
    ray_tracing: bool = True
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

    scene_manager = SapienSceneManager(
        cfg.sim_freq, ray_tracing=cfg.ray_tracing
    )
    _ = scene_manager.initialize_circular_cameras(
        num_cameras=cfg.num_cameras,
        radius=cfg.camera_radius,
        height=cfg.camera_height,
        target_pt=cfg.camera_target_pt,
        image_hw=cfg.image_hw,
        fovy_deg=cfg.fovy_deg,
    )
    with open(cfg.layout_path, "r") as f:
        layout_data: LayoutInfo = LayoutInfo.from_dict(json.load(f))

    actors = load_assets_from_layout_file(
        scene_manager.scene,
        cfg.layout_path,
        cfg.z_offset,
    )
    agent = load_mani_skill_robot(
        scene_manager.scene, cfg.layout_path, cfg.control_freq
    )

    frames = defaultdict(list)
    image_cnt = 0
    for step in tqdm(range(cfg.sim_step), desc="Simulation"):
        scene_manager.scene.step()
        agent.reset(agent.init_qpos)
        if step % cfg.render_interval != 0:
            continue
        scene_manager.scene.update_render()
        image_cnt += 1
        for camera in scene_manager.cameras:
            camera.take_picture()
            images = render_images(camera, cfg.render_keys)
            frames[camera.name].append(images)

    actions = dict()
    if cfg.insert_robot:
        grasper = FrankaPandaGrasper(
            agent,
            cfg.control_freq,
        )
        for node in layout_data.relation[
            Scene3DItemEnum.MANIPULATED_OBJS.value
        ]:
            actions[node] = grasper.compute_grasp_action(
                actor=actors[node], reach_target_only=True
            )

    if "Foreground" not in cfg.render_keys:
        return

    asset_root = os.path.dirname(cfg.layout_path)
    bg_node = layout_data.relation[Scene3DItemEnum.BACKGROUND.value]
    gs_path = f"{asset_root}/{layout_data.assets[bg_node]}/gs_model.ply"
    gs_model: GaussianOperator = GaussianOperator.load_from_ply(gs_path)
    x, y, z, qx, qy, qz, qw = layout_data.position[bg_node]
    qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw], cfg.init_quat)
    init_pose = torch.tensor([x, y, z, qx, qy, qz, qw])
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
    for idx, camera in enumerate(scene_manager.cameras):
        # Scene rendering
        if idx == 0:
            for step in range(image_cnt):
                rgba = alpha_blend_rgba(
                    frames[camera.name][step]["Foreground"],
                    bg_images[camera.name],
                )
                video_frames.append(np.array(rgba))

        # Grasp rendering
        for node in actions:
            if actions[node] is None:
                continue
            logger.info(f"Render SIM grasping in camera {idx} for {node}...")
            for action in actions[node]:
                grasp_frames = scene_manager.step_action(
                    agent,
                    torch.Tensor(action[None, ...]),
                    scene_manager.cameras,
                    cfg.render_keys,
                    sim_steps_per_control=cfg.sim_freq // cfg.control_freq,
                )
                rgba = alpha_blend_rgba(
                    grasp_frames[camera.name][0]["Foreground"],
                    bg_images[camera.name],
                )
                video_frames.append(np.array(rgba))

            agent.reset(agent.init_qpos)

    os.makedirs(cfg.output_dir, exist_ok=True)
    video_path = f"{cfg.output_dir}/Iscene.mp4"
    imageio.mimsave(video_path, video_frames, fps=30)
    logger.info(f"Interative 3D Scene Visualization saved in {video_path}")


if __name__ == "__main__":
    entrypoint()
