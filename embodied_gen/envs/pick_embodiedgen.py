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

import numpy as np
import sapien
import torch
import torchvision.transforms as transforms
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
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
    """PickEmbodiedGen as gym env example for object pick-and-place tasks.

    This environment simulates a robot interacting with 3D assets in the
    embodiedgen generated scene in SAPIEN. It supports multi-environment setups,
    dynamic reconfiguration, and hybrid rendering with 3D Gaussian Splatting.

    Example:
        Use `gym.make` to create the `PickEmbodiedGen-v1` parallel environment.
        ```python
        import gymnasium as gym
        env = gym.make(
            "PickEmbodiedGen-v1",
            num_envs=cfg.num_envs,
            render_mode=cfg.render_mode,
            enable_shadow=cfg.enable_shadow,
            layout_file=cfg.layout_file,
            control_mode=cfg.control_mode,
            camera_cfg=dict(
                camera_eye=cfg.camera_eye,
                camera_target_pt=cfg.camera_target_pt,
                image_hw=cfg.image_hw,
                fovy_deg=cfg.fovy_deg,
            ),
        )
        ```
    """

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
        """Initializes the PickEmbodiedGen environment.

        Args:
            *args: Variable length argument list for the base class.
            robot_uids: The robot(s) to use in the environment.
            robot_init_qpos_noise: Noise added to the robot's initial joint
                positions.
            num_envs: The number of parallel environments to create.
            reconfiguration_freq: How often to reconfigure the scene. If None,
                it is set based on num_envs.
            **kwargs: Additional keyword arguments for environment setup,
                including layout_file, replace_objs, enable_grasp, etc.
        """
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
        self.init_3dgs_quat = kwargs.pop(
            "init_3dgs_quat", [0.7071, 0, 0, 0.7071]
        )
        # Add small offset in z-axis to avoid collision.
        self.objs_z_offset = kwargs.pop("objs_z_offset", 0.002)
        self.robot_z_offset = kwargs.pop("robot_z_offset", 0.002)
        self.camera_cfg = kwargs.pop("camera_cfg", None)
        if self.camera_cfg is None:
            self.camera_cfg = dict(
                camera_eye=[0.9, 0.0, 1.1],
                camera_target_pt=[0.0, 0.0, 0.9],
                image_hw=[256, 256],
                fovy_deg=75,
            )

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
                self.layouts, num_envs, self.init_3dgs_quat
            )

    @staticmethod
    def init_env_layouts(
        layout_file: str, num_envs: int, replace_objs: bool
    ) -> list[LayoutInfo]:
        """Initializes and saves layout files for each environment instance.

        For each environment, this method creates a layout configuration. If
        `replace_objs` is True, it generates new object placements for each
        subsequent environment. The generated layouts are saved as new JSON
        files.

        Args:
            layout_file: Path to the base layout JSON file.
            num_envs: The number of environments to create layouts for.
            replace_objs: If True, generates new object placements for each
                environment after the first one using BFS placement.

        Returns:
            A list of file paths to the generated layout for each environment.
        """
        layouts = []
        for env_idx in range(num_envs):
            if replace_objs and env_idx > 0:
                layout_info = bfs_placement(layout_file)
            else:
                layout_info = json.load(open(layout_file, "r"))
                layout_info = LayoutInfo.from_dict(layout_info)

            layout_path = layout_file.replace(".json", f"_env{env_idx}.json")
            with open(layout_path, "w") as f:
                json.dump(layout_info.to_dict(), f, indent=4)

            layouts.append(layout_path)

        return layouts

    @staticmethod
    def compute_robot_init_pose(
        layouts: list[str], num_envs: int, z_offset: float = 0.0
    ) -> list[list[float]]:
        """Computes the initial pose for the robot in each environment.

        Args:
            layouts: A list of file paths to the environment layouts.
            num_envs: The number of environments.
            z_offset: An optional vertical offset to apply to the robot's
                position to prevent collisions.

        Returns:
            A list of initial poses ([x, y, z, qw, qx, qy, qz]) for the robot
            in each environment.
        """
        robot_pose = []
        for env_idx in range(num_envs):
            layout = json.load(open(layouts[env_idx], "r"))
            layout = LayoutInfo.from_dict(layout)
            robot_node = layout.relation[Scene3DItemEnum.ROBOT.value]
            x, y, z, qx, qy, qz, qw = layout.position[robot_node]
            robot_pose.append([x, y, z + z_offset, qw, qx, qy, qz])

        return robot_pose

    @property
    def _default_sim_config(self):
        """Returns the default simulation configuration.

        Returns:
            The default simulation configuration object.
        """
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
        """Returns the default sensor configurations for the agent.

        Returns:
            A list containing the default camera configuration.
        """
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])

        return [
            CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        """Returns the default camera configuration for human-friendly rendering.

        Returns:
            The default camera configuration for the renderer.
        """
        pose = sapien_utils.look_at(
            eye=self.camera_cfg["camera_eye"],
            target=self.camera_cfg["camera_target_pt"],
        )

        return CameraConfig(
            "render_camera",
            pose,
            self.camera_cfg["image_hw"][1],
            self.camera_cfg["image_hw"][0],
            np.deg2rad(self.camera_cfg["fovy_deg"]),
            0.01,
            100,
        )

    def _load_agent(self, options: dict):
        """Loads the agent (robot) and a ground plane into the scene.

        Args:
            options: A dictionary of options for loading the agent.
        """
        self.ground = build_ground(self.scene)
        super()._load_agent(options, sapien.Pose(p=[-10, 0, 10]))

    def _load_scene(self, options: dict):
        """Loads all assets, objects, and the goal site into the scene.

        This method iterates through the layouts for each environment, loads the
        specified assets, and adds them to the simulation. It also creates a
        kinematic sphere to represent the goal site.

        Args:
            options: A dictionary of options for loading the scene.
        """
        all_objects = []
        logger.info(f"Loading EmbodiedGen assets...")
        for env_idx in range(self.num_envs):
            env_actors = load_assets_from_layout_file(
                self.scene,
                self.layouts[env_idx],
                z_offset=self.objs_z_offset,
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
        """Initializes an episode for a given set of environments.

        This method sets the goal position, resets the robot's joint positions
        with optional noise, and sets its root pose.

        Args:
            env_idx: A tensor of environment indices to initialize.
            options: A dictionary of options for initialization.
        """
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
        self, layouts: list[str], num_envs: int, init_quat: list[float]
    ) -> dict[str, np.ndarray]:
        """Renders background images using a pre-trained Gaussian Splatting model.

        This method pre-renders the static background for each environment from
        the perspective of all cameras to be used for hybrid rendering.

        Args:
            layouts: A list of file paths to the environment layouts.
            num_envs: The number of environments.
            init_quat: An initial quaternion to orient the Gaussian Splatting
                model.

        Returns:
            A dictionary mapping a unique key (e.g., 'camera-env_idx') to the
            rendered background image as a numpy array.
        """
        sim_coord_align = (
            torch.tensor(SIM_COORD_ALIGN).to(torch.float32).to(self.device)
        )
        cameras = self.scene.sensors.copy()
        cameras.update(self.scene.human_render_cameras)

        # Preload the background Gaussian Splatting model.
        asset_root = os.path.dirname(layouts[0])
        layout = LayoutInfo.from_dict(json.load(open(layouts[0], "r")))
        bg_node = layout.relation[Scene3DItemEnum.BACKGROUND.value]
        gs_path = os.path.join(
            asset_root, layout.assets[bg_node], "gs_model.ply"
        )
        raw_gs: GaussianOperator = GaussianOperator.load_from_ply(gs_path)
        bg_images = dict()
        for env_idx in tqdm(range(num_envs), desc="Pre-rendering Background"):
            layout = json.load(open(layouts[env_idx], "r"))
            layout = LayoutInfo.from_dict(layout)
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
        """Renders the environment based on the configured render_mode.

        Raises:
            RuntimeError: If `render_mode` is not set.
            NotImplementedError: If the `render_mode` is not supported.

        Returns:
            The rendered output, which varies depending on the render mode.
        """
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
        """Renders an RGB image from the human-facing render camera.

        Args:
            camera_name: The name of the camera to render from. If None, uses
                all human render cameras.
            return_alpha: Whether to include the alpha channel in the output.

        Returns:
            A numpy array representing the rendered image(s). If multiple
            cameras are used, the images are tiled.
        """
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
        """Renders images from all on-board sensor cameras.

        Returns:
            A tiled image of all sensor outputs as a numpy array.
        """
        images = []
        sensor_images = self.get_sensor_images()
        for image in sensor_images.values():
            for img in image.values():
                images.append(img)
        return tile_images(images)

    def hybrid_render(self):
        """Renders a hybrid image by blending simulated foreground with a background.

        The foreground is rendered with an alpha channel and then blended with
        the pre-rendered Gaussian Splatting background image.

        Returns:
            A torch tensor of the final blended RGB images.
        """
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
        """Evaluates the current state of the environment.

        Checks for task success criteria such as whether the object is grasped,
        placed at the goal, and if the robot is static.

        Returns:
            A dictionary containing boolean tensors for various success
            metrics, including 'is_grasped', 'is_obj_placed', and overall
            'success'.
        """
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
        """Gets extra information for the observation dictionary.

        Args:
            info: A dictionary containing evaluation information.

        Returns:
            An empty dictionary, as no extra observations are added.
        """

        return dict()

    def compute_dense_reward(self, obs: any, action: torch.Tensor, info: dict):
        """Computes a dense reward for the current step.

        The reward is a composite of reaching, grasping, placing, and
        maintaining a static final pose.

        Args:
            obs: The current observation.
            action: The action taken in the current step.
            info: A dictionary containing evaluation information from `evaluate()`.

        Returns:
            A tensor containing the dense reward for each environment.
        """
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
        """Computes a dense reward normalized to be between 0 and 1.

        Args:
            obs: The current observation.
            action: The action taken in the current step.
            info: A dictionary containing evaluation information from `evaluate()`.

        Returns:
            A tensor containing the normalized dense reward for each environment.
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
