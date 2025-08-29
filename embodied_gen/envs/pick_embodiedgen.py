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
from copy import deepcopy

import numpy as np
import sapien
import torch
import torchvision.transforms as transforms
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import (
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from mani_skill.utils.visualization.misc import tile_images
from tqdm import tqdm
from embodied_gen.models.gs_model import GaussianOperator
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.geometry import bfs_placement, quaternion_multiply
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import alpha_blend_rgba
from embodied_gen.utils.simulation import (
    SIM_COORD_ALIGN,
    load_assets_from_layout_file,
)

__all__ = ["PickEmbodiedGen"]


@register_env("PickEmbodiedGen-v1", max_episode_steps=100)
class PickEmbodiedGen(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    goal_thresh = 0.0

    def __init__(
        self,
        *args,
        robot_uids: str | list[str] = "panda",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq: int = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0

        # Init params from kwargs.
        layout_file = kwargs.pop("layout_file", None)
        replace_objs = kwargs.pop("replace_objs", True)
        self.enable_grasp = kwargs.pop("enable_grasp", False)
        self.init_quat = kwargs.pop("init_quat", [0.7071, 0, 0, 0.7071])
        # Add small offset in z-axis to avoid collision.
        self.objs_z_offset = kwargs.pop("objs_z_offset", 0.002)
        self.robot_z_offset = kwargs.pop("robot_z_offset", 0.002)

        self.layouts = self.init_env_layouts(
            layout_file, num_envs, replace_objs
        )
        self.robot_pose = self.compute_robot_init_pose(
            self.layouts, num_envs, self.robot_z_offset
        )
        self.env_actors = dict()
        self.image_transform = transforms.PILToTensor()

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

        self.bg_images = dict()
        if self.render_mode == "hybrid":
            self.bg_images = self.render_gs3d_images(
                self.layouts, num_envs, self.init_quat
            )

    @staticmethod
    def init_env_layouts(
        layout_file: str, num_envs: int, replace_objs: bool
    ) -> list[LayoutInfo]:
        layout = LayoutInfo.from_dict(json.load(open(layout_file, "r")))
        layouts = []
        for env_idx in range(num_envs):
            if replace_objs and env_idx > 0:
                layout = bfs_placement(deepcopy(layout))
            layouts.append(layout)

        return layouts

    @staticmethod
    def compute_robot_init_pose(
        layouts: list[LayoutInfo], num_envs: int, z_offset: float = 0.0
    ) -> list[list[float]]:
        robot_pose = []
        for env_idx in range(num_envs):
            layout = layouts[env_idx]
            robot_node = layout.relation[Scene3DItemEnum.ROBOT.value]
            x, y, z, qx, qy, qz, qw = layout.position[robot_node]
            robot_pose.append([x, y, z + z_offset, qw, qx, qy, qz])

        return robot_pose

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=30,
                # contact_offset=0.04,
                # rest_offset=0.001,
            ),
            # sim_freq=200,
            control_freq=50,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])

        return [
            CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.9, 0.0, 1.1], target=[0.0, 0.0, 0.9]
        )

        return CameraConfig(
            "render_camera", pose, 256, 256, np.deg2rad(75), 0.01, 100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-10, 0, 10]))

    def _load_scene(self, options: dict):
        all_objects = []
        logger.info(f"Loading assets and decomposition mesh collisions...")
        for env_idx in range(self.num_envs):
            env_actors = load_assets_from_layout_file(
                self.scene,
                self.layouts[env_idx],
                z_offset=self.objs_z_offset,
                init_quat=self.init_quat,
                env_idx=env_idx,
            )
            self.env_actors[f"env{env_idx}"] = env_actors
            all_objects.extend(env_actors.values())

        self.obj = all_objects[-1]
        for obj in all_objects:
            self.remove_from_state_dict_registry(obj)

        self.all_objects = Actor.merge(all_objects, name="all_objects")
        self.add_to_state_dict_registry(self.all_objects)

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 3 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                np.random.normal(
                    0, self.robot_init_qpos_noise, (self.num_envs, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_root_pose(np.array(self.robot_pose))
            self.agent.reset(qpos)
            self.agent.init_qpos = qpos
            self.agent.controller.controllers["gripper"].reset()

    def render_gs3d_images(
        self, layouts: list[LayoutInfo], num_envs: int, init_quat: list[float]
    ) -> dict[str, np.ndarray]:
        sim_coord_align = (
            torch.tensor(SIM_COORD_ALIGN).to(torch.float32).to(self.device)
        )
        cameras = self.scene.sensors.copy()
        cameras.update(self.scene.human_render_cameras)

        bg_node = layouts[0].relation[Scene3DItemEnum.BACKGROUND.value]
        gs_path = os.path.join(layouts[0].assets[bg_node], "gs_model.ply")
        raw_gs: GaussianOperator = GaussianOperator.load_from_ply(gs_path)
        bg_images = dict()
        for env_idx in tqdm(range(num_envs), desc="Pre-rendering Background"):
            layout = layouts[env_idx]
            x, y, z, qx, qy, qz, qw = layout.position[bg_node]
            qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw], init_quat)
            init_pose = torch.tensor([x, y, z, qx, qy, qz, qw])
            gs_model = raw_gs.get_gaussians(instance_pose=init_pose)
            for key in cameras:
                camera = cameras[key]
                Ks = camera.camera.get_intrinsic_matrix()  # (n_env, 3, 3)
                c2w = camera.camera.get_model_matrix()  # (n_env, 4, 4)
                result = gs_model.render(
                    c2w[env_idx] @ sim_coord_align,
                    Ks[env_idx],
                    image_width=camera.config.width,
                    image_height=camera.config.height,
                )
                bg_images[f"{key}-env{env_idx}"] = result.rgb[..., ::-1]

        return bg_images

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render_mode is not set.")
        if self.render_mode == "human":
            return self.render_human()
        elif self.render_mode == "rgb_array":
            res = self.render_rgb_array()
            return res
        elif self.render_mode == "sensors":
            res = self.render_sensors()
            return res
        elif self.render_mode == "all":
            return self.render_all()
        elif self.render_mode == "hybrid":
            return self.hybrid_render()
        else:
            raise NotImplementedError(
                f"Unsupported render mode {self.render_mode}."
            )

    def render_rgb_array(
        self, camera_name: str = None, return_alpha: bool = False
    ):
        for obj in self._hidden_objects:
            obj.show_visual()
        self.scene.update_render(
            update_sensors=False, update_human_render_cameras=True
        )
        images = []
        render_images = self.scene.get_human_render_camera_images(
            camera_name, return_alpha
        )
        for image in render_images.values():
            images.append(image)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        for obj in self._hidden_objects:
            obj.hide_visual()
        return tile_images(images)

    def render_sensors(self):
        images = []
        sensor_images = self.get_sensor_images()
        for image in sensor_images.values():
            for img in image.values():
                images.append(img)
        return tile_images(images)

    def hybrid_render(self):
        fg_images = self.render_rgb_array(
            return_alpha=True
        )  # (n_env, h, w, 3)
        images = []
        for key in self.bg_images:
            if "render_camera" not in key:
                continue
            env_idx = int(key.split("-env")[-1])
            rgba = alpha_blend_rgba(
                fg_images[env_idx].cpu().numpy(), self.bg_images[key]
            )
            images.append(self.image_transform(rgba))

        images = torch.stack(images, dim=0)
        images = images.permute(0, 2, 3, 1)

        return images[..., :3]

    def evaluate(self):
        obj_to_goal_pos = (
            self.obj.pose.p
        )  # self.goal_site.pose.p - self.obj.pose.p
        is_obj_placed = (
            torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)

        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=self.agent.is_grasping(self.obj),
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )

    def _get_obs_extra(self, info: dict):

        return dict()

    def compute_dense_reward(self, obs: any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        # obj_to_goal_dist = torch.linalg.norm(
        #     self.goal_site.pose.p - self.obj.pose.p, axis=1
        # )
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5
            * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
