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
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Literal

import mplib
import numpy as np
import sapien.core as sapien
import torch
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from embodied_gen.data.utils import DiffrastRender
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.geometry import quaternion_multiply

logger = logging.getLogger(__name__)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
)
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

COLORMAP = list(set(ImageColor.colormap.values()))
COLOR_PALETTE = np.array(
    [ImageColor.getrgb(c) for c in COLORMAP], dtype=np.uint8
)
SIM_COORD_ALIGN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)  # Used to align SAPIEN, MuJoCo coordinate system with the world coordinate system

__all__ = [
    "SIM_COORD_ALIGN",
    "FrankaPandaGrasper",
    "SimpleGrasper",
]


class SapienSceneManager:
    """A class to manage SAPIEN simulator."""

    def __init__(
        self, sim_freq: int, ray_tracing: bool, device: str = "cuda"
    ) -> None:
        self.sim_freq = sim_freq
        self.ray_tracing = ray_tracing
        self.device = device
        self.renderer = sapien.SapienRenderer()
        self.scene = self._setup_scene()
        self.cameras: list[sapien.render.RenderCameraComponent] = []
        self.actors: dict[str, sapien.pysapien.Entity] = {}

    def _setup_scene(self) -> sapien.Scene:
        """Set up the SAPIEN scene with lighting and ground."""
        # Ray tracing settings
        if self.ray_tracing:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(10)
            sapien.render.set_ray_tracing_denoiser("oidn")

        # TODO: Support GPU.
        # import sapien.physx as physx
        # physx.enable_gpu()
        # scene = sapien.Scene(systems=[sapien.physx.PhysxGpuSystem(), sapien.render.RenderSystem()])

        scene = sapien.Scene()
        scene.set_timestep(1 / self.sim_freq)

        # Add lighting
        scene.set_ambient_light([0.2, 0.2, 0.2])
        scene.add_directional_light(
            direction=[0, 1, -1],
            color=[1.5, 1.45, 1.4],
            shadow=True,
            shadow_map_size=2048,
        )
        scene.add_directional_light(
            direction=[0, -0.5, 1], color=[0.8, 0.8, 0.85], shadow=False
        )
        scene.add_directional_light(
            direction=[0, -1, 1], color=[1.0, 1.0, 1.0], shadow=False
        )

        ground_material = self.renderer.create_material()
        ground_material.base_color = [0.5, 0.5, 0.5, 1]  # rgba, gray
        ground_material.roughness = 0.7
        ground_material.metallic = 0.0
        scene.add_ground(0, render_material=ground_material)

        return scene

    def step_action(
        self,
        action: torch.Tensor,
        sim_steps_per_control: int,
        cameras: list[sapien.render.RenderCameraComponent],
        render_keys: list[str],
    ) -> dict:
        self.robot.set_action(action)
        frames = defaultdict(list)
        for _ in range(sim_steps_per_control):
            self.scene.step()

        self.scene.update_render()
        for camera in cameras:
            camera.take_picture()
            images = self.render_images(camera, render_keys=render_keys)
            frames[camera.name].append(images)

        return frames

    def load_exists_robot(
        self,
        robot_name: str,
        pose: sapien.Pose,
        control_freq: int = 20,
        robot_init_qpos_noise: float = 0.02,
        control_mode: str = "pd_joint_pos",
        backend_str: tuple[str, str] = ("cpu", "gpu"),
    ):
        from mani_skill.agents import REGISTERED_AGENTS
        from mani_skill.envs.scene import ManiSkillScene
        from mani_skill.envs.utils.system.backend import (
            parse_sim_and_render_backend,
        )

        if robot_name == "franka":
            robot_name = "panda"  # remapping
        if robot_name not in REGISTERED_AGENTS:
            raise KeyError(
                f"Robot `{robot_name}` not registered, chosen from {REGISTERED_AGENTS.keys()}."
            )

        ROBOT_CLS = REGISTERED_AGENTS[robot_name].agent_cls
        # backend = parse_sim_and_render_backend("physx_cuda", "gpu")
        backend = parse_sim_and_render_backend(*backend_str)
        scene = ManiSkillScene(
            [self.scene], device=backend_str[0], backend=backend
        )
        robot = ROBOT_CLS(
            scene=scene,
            control_freq=control_freq,
            control_mode=control_mode,
            initial_pose=pose,
        )
        # Set robot init joint rad agree(joint0 to joint6 w 2 finger).
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
                0, robot_init_qpos_noise, (len(scene.sub_scenes), len(qpos))
            )
            + qpos
        )
        qpos[:, -2:] = 0.04
        robot.reset(qpos)
        robot.init_qpos = robot.robot.qpos
        robot.controller.controllers["gripper"].reset()

        return robot

    def load_actor_from_urdf(
        self,
        file_path: str,
        pose: sapien.Pose,
        use_static: bool = False,
        update_mass: bool = False,
    ) -> sapien.pysapien.Entity:
        """Load an actor from a URDF file into the scene.

        Args:
            file_path (str): Path to the URDF file.
            pose (sapien.Pose): Pose of the actor.
            use_static (bool): Whether to create a static actor.
            update_mass (bool): Whether to update the actor's mass from the URDF.

        Returns:
            sapien.pysapien.Entity: The loaded actor.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        node_name = root.get("name")
        file_dir = os.path.dirname(file_path)
        visual_file = root.find('.//visual/geometry/mesh').get("filename")
        collision_file = root.find('.//collision/geometry/mesh').get(
            "filename"
        )
        visual_file = os.path.join(file_dir, visual_file)
        collision_file = os.path.join(file_dir, collision_file)
        static_fric = root.find('.//collision/gazebo/mu1').text
        dynamic_fric = root.find('.//collision/gazebo/mu2').text

        material = self.scene.create_physical_material(
            static_friction=np.clip(float(static_fric), 0.1, 0.7),
            dynamic_friction=np.clip(float(dynamic_fric), 0.1, 0.6),
            restitution=0.05,
        )
        builder = self.scene.create_actor_builder()
        body_type = "static" if use_static else "dynamic"
        builder.set_physx_body_type(body_type)
        builder.add_multiple_convex_collisions_from_file(
            collision_file, material=material
        )
        builder.add_visual_from_file(visual_file)
        actor = builder.build(name=node_name)
        actor.set_name(node_name)
        actor.set_pose(pose)

        if update_mass and hasattr(actor.components[1], "mass"):
            node_mass = float(root.find('.//inertial/mass').get("value"))
            actor.components[1].set_mass(node_mass)

        return actor

    def create_camera(
        self,
        cam_name: str,
        pose: sapien.Pose,
        image_hw: tuple[int, int],
        fovy_deg: float,
    ) -> sapien.render.RenderCameraComponent:
        """Create a single camera in the scene.

        Args:
            cam_name (str): Name of the camera.
            pose (sapien.Pose): Camera pose p=(x, y, z), q=(w, x, y, z)
            image_hw (Tuple[int, int]): Image resolution (height, width) for cameras.
            fovy_deg (float): Field of view in degrees for cameras.

        Returns:
            sapien.render.RenderCameraComponent: The created camera.
        """
        cam_actor = self.scene.create_actor_builder().build_kinematic()
        cam_actor.set_pose(pose)
        camera = self.scene.add_mounted_camera(
            name=cam_name,
            mount=cam_actor,
            pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
            width=image_hw[1],
            height=image_hw[0],
            fovy=np.deg2rad(fovy_deg),
            near=0.01,
            far=100,
        )
        self.cameras.append(camera)

        return camera

    def initialize_circular_cameras(
        self,
        num_cameras: int,
        radius: float,
        height: float,
        target_pt: list[float],
        image_hw: tuple[int, int],
        fovy_deg: float,
    ) -> list[sapien.render.RenderCameraComponent]:
        """Initialize multiple cameras arranged in a circle.

        Args:
            num_cameras (int): Number of cameras to create.
            radius (float): Radius of the camera circle.
            height (float): Fixed Z-coordinate of the cameras.
            target_pt (list[float]): 3D point (x, y, z) that cameras look at.
            image_hw (Tuple[int, int]): Image resolution (height, width) for cameras.
            fovy_deg (float): Field of view in degrees for cameras.

        Returns:
            List[sapien.render.RenderCameraComponent]: List of created cameras.
        """
        angle_step = 2 * np.pi / num_cameras
        world_up_vec = np.array([0.0, 0.0, 1.0])
        target_pt = np.array(target_pt)

        for i in range(num_cameras):
            angle = i * angle_step
            cam_x = radius * np.cos(angle)
            cam_y = radius * np.sin(angle)
            cam_z = height
            eye_pos = [cam_x, cam_y, cam_z]

            forward_vec = target_pt - eye_pos
            forward_vec = forward_vec / np.linalg.norm(forward_vec)
            temp_right_vec = np.cross(forward_vec, world_up_vec)

            if np.linalg.norm(temp_right_vec) < 1e-6:
                temp_right_vec = np.array([1.0, 0.0, 0.0])
                if np.abs(np.dot(temp_right_vec, forward_vec)) > 0.99:
                    temp_right_vec = np.array([0.0, 1.0, 0.0])

            right_vec = temp_right_vec / np.linalg.norm(temp_right_vec)
            up_vec = np.cross(right_vec, forward_vec)
            rotation_matrix = np.array([forward_vec, -right_vec, up_vec]).T

            rot = R.from_matrix(rotation_matrix)
            scipy_quat = rot.as_quat()  # (x, y, z, w)
            quat = [
                scipy_quat[3],
                scipy_quat[0],
                scipy_quat[1],
                scipy_quat[2],
            ]  # (w, x, y, z)

            self.create_camera(
                f"camera_{i}",
                sapien.Pose(p=eye_pos, q=quat),
                image_hw,
                fovy_deg,
            )

        return self.cameras

    def render_images(
        self,
        camera: sapien.render.RenderCameraComponent,
        render_keys: list[
            Literal[
                "Color",
                "Segmentation",
                "Normal",
                "Mask",
                "Depth",
                "Foreground",
            ]
        ] = None,
    ) -> dict[str, Image.Image]:
        """Render images from a given camera.

        Args:
            camera (sapien.render.RenderCameraComponent): The camera to render from.
            render_keys (List[str]): Types of images to render (e.g., Color, Segmentation).

        Returns:
            Dict[str, Image.Image]: Dictionary of rendered images.
        """
        if render_keys is None:
            render_keys = [
                "Color",
                "Segmentation",
                "Normal",
                "Mask",
                "Depth",
                "Foreground",
            ]

        results: dict[str, Image.Image] = {}
        if "Color" in render_keys:
            color = camera.get_picture("Color")
            color_rgb = (np.clip(color[..., :3], 0, 1) * 255).astype(np.uint8)
            results["Color"] = Image.fromarray(color_rgb)

        if "Mask" in render_keys:
            alpha = (np.clip(color[..., 3], 0, 1) * 255).astype(np.uint8)
            results["Mask"] = Image.fromarray(alpha)

        if "Segmentation" in render_keys:
            seg_labels = camera.get_picture("Segmentation")
            label0 = seg_labels[..., 0].astype(np.uint8)
            seg_color = self.COLOR_PALETTE[label0]
            results["Segmentation"] = Image.fromarray(seg_color)

        if "Foreground" in render_keys:
            seg_labels = camera.get_picture("Segmentation")
            label0 = seg_labels[..., 0]
            mask = np.where((label0 > 1), 255, 0).astype(np.uint8)
            color = camera.get_picture("Color")
            color_rgb = (np.clip(color[..., :3], 0, 1) * 255).astype(np.uint8)
            foreground = np.concatenate([color_rgb, mask[..., None]], axis=-1)
            results["Foreground"] = Image.fromarray(foreground)

        if "Normal" in render_keys:
            normal = camera.get_picture("Normal")[..., :3]
            normal_img = (((normal + 1) / 2) * 255).astype(np.uint8)
            results["Normal"] = Image.fromarray(normal_img)

        if "Depth" in render_keys:
            position_map = camera.get_picture("Position")
            depth = -position_map[..., 2]
            alpha = torch.tensor(color[..., 3], dtype=torch.float32)
            norm_depth = DiffrastRender.normalize_map_by_mask(
                torch.tensor(depth), alpha
            )
            depth_img = (norm_depth * 255).to(torch.uint8).numpy()
            results["Depth"] = Image.fromarray(depth_img)

        return results

    def load_assets_from_layout_file(
        self,
        layout: LayoutInfo,
        z_offset: float = 0.0,
        init_quat: list[float] = [0, 0, 0, 1],
        robot_name: str = None,
        control_freq: int = 40,
        robot_init_qpos_noise: float = 0,
    ) -> None:
        """Load assets from `EmbodiedGen` layout-gen output and create actors in the scene.

        Args:
            layout (LayoutInfo): The layout information data.
            z_offset (float): Offset to apply to the Z-coordinate of non-context objects.
            init_quat (List[float]): Initial quaternion (x, y, z, w) for orientation adjustment.
            robot_name (str): Name of the robot to load, None not load robot.
            control_freq (int): Control frequency for the robot, default is 40.
            robot_init_qpos_noise (float): Noise to add to the robot's initial joint positions, default is 0.
        """
        for node in layout.assets:
            file_dir = layout.assets[node]
            file_name = f"{node.replace(' ', '_')}.urdf"
            urdf_file = os.path.join(file_dir, file_name)

            if layout.objs_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
                continue

            position = layout.position[node].copy()
            if layout.objs_mapping[node] != Scene3DItemEnum.CONTEXT.value:
                position[2] += z_offset

            use_static = (
                layout.relation.get(Scene3DItemEnum.CONTEXT.value, None)
                == node
            )

            # Combine initial quaternion with object quaternion
            x, y, z, qx, qy, qz, qw = position
            qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw], init_quat)
            actor = self.load_actor_from_urdf(
                urdf_file,
                sapien.Pose(p=[x, y, z], q=[qw, qx, qy, qz]),
                use_static=use_static,
                update_mass=False,
            )
            self.actors[node] = actor

        if robot_name is not None:
            robot_node = layout.relation[Scene3DItemEnum.ROBOT.value]
            x, y, z, qx, qy, qz, qw = layout.position[robot_node]
            delta_z = 0.002  # Add small offset to avoid collision.
            pose = sapien.Pose([x, y, z + delta_z], [qw, qx, qy, qz])
            self.robot = self.load_exists_robot(
                robot_name, pose, control_freq, robot_init_qpos_noise
            )

        return


class FrankaPandaGrasper(object):
    """Grasper planner for Franka Panda robot arm.

    Args:
        robot: See from `mani_skill.agents.robots.panda.panda`.
        scene: The `SapienSceneManager` object.
        sim_steps_per_control: Number of simulation steps per control step.
        control_timestep: Time step for planning.
        joint_vel_limits: Velocity limit for all joints (default: 0.9)
        joint_acc_limits: Acceleration limit for all joints (default: 0.9)
        finger_length: Length of the robot's fingers (default: 0.025)
    """

    def __init__(
        self,
        robot,
        scene: SapienSceneManager,
        sim_steps_per_control: int,
        control_timestep: float,
        joint_vel_limits: float = 0.9,
        joint_acc_limits: float = 0.9,
        finger_length: float = 0.025,
    ) -> None:
        self.agent = robot
        self.robot = robot.robot
        self.scene = scene
        self.sim_steps_per_control = sim_steps_per_control
        self.control_timestep = control_timestep
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.finger_length = finger_length
        self.planner = self._setup_planner()

    def _setup_planner(self) -> mplib.Planner:
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [
            joint.get_name() for joint in self.robot.get_active_joints()
        ]

        planner = mplib.Planner(
            urdf=self.agent.urdf_path,
            srdf=self.agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(self.robot.pose.raw_pose[0].tolist())

        return planner

    def update_state(
        self,
        result: dict,
        gripper_state: Literal[-1, 1],
        sim_steps_per_control: int,
        cameras: list[sapien.render.RenderCameraComponent],
        render_keys: list[str],
    ) -> dict[str, list[Image.Image]]:
        n_step = len(result["position"])
        total_frames = defaultdict(list)
        for i in tqdm(range(n_step), desc="Grasping"):
            qpos = result["position"][min(i, n_step - 1)]
            action = np.hstack([qpos, gripper_state])[None, ...]
            action = torch.from_numpy(action).float()
            frames = self.scene.step_action(
                action, sim_steps_per_control, cameras, render_keys
            )
            for camera in cameras:
                total_frames[camera.name].extend(frames[camera.name])

        return total_frames

    def control_gripper(
        self,
        sim_steps_per_control: int,
        cameras: list[sapien.render.RenderCameraComponent],
        render_keys: list[str],
        gripper_state: Literal[-1, 1],
        n_step: int = 10,
    ) -> dict[str, list[Image.Image]]:
        total_frames = defaultdict(list)
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for _ in range(n_step):
            action = np.hstack([qpos, gripper_state])[None, ...]
            action = torch.from_numpy(action).float()
            frames = self.scene.step_action(
                action, sim_steps_per_control, cameras, render_keys
            )
            for camera in cameras:
                total_frames[camera.name].extend(frames[camera.name])

        return total_frames

    def move_to_pose(
        self,
        pose: sapien.Pose,
        sim_steps_per_control: int,
        control_timestep: float,
        cameras: list[sapien.render.RenderCameraComponent],
        render_keys: list[str],
        gripper_state: Literal[-1, 1],
        use_point_cloud: bool = False,
        n_max_step: int = 100,
    ) -> dict[str, list[Image.Image]]:
        # First try screw motion planning
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=control_timestep,
            use_point_cloud=use_point_cloud,
        )

        # Fallback to standard planning if screw fails
        if result["status"] != "Success":
            result = self.planner.plan_qpos_to_pose(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=control_timestep,
                use_point_cloud=use_point_cloud,
            )

        if result["status"] != "Success":
            return defaultdict(list)

        for key in result:
            if key in ["status", "duration"]:
                continue
            sample_ratio = (len(result[key]) // n_max_step) + 1
            result[key] = result[key][::sample_ratio]

        frames = self.update_state(
            result, gripper_state, sim_steps_per_control, cameras, render_keys
        )

        return frames

    def render_grasp(
        self,
        actor: sapien.pysapien.Entity,
        cameras: list[sapien.render.RenderCameraComponent],
        render_keys: list[
            Literal[
                "Color",
                "Foreground",
                "Segmentation",
                "Normal",
                "Mask",
                "Depth",
            ]
        ],
        reach_target_only: bool = True,
    ) -> dict[str, list[Image.Image]]:
        physx_rigid = actor.components[1]
        mesh = get_component_mesh(physx_rigid, to_world_frame=True)
        obb = mesh.bounding_box_oriented
        approaching = np.array([0, 0, -1])
        target_closing = (
            self.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
            .cpu()
            .numpy()
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=self.finger_length,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        raw_tcp_pose = self.agent.tcp.pose.sp
        grasp_pose = self.agent.build_grasp_pose(approaching, closing, center)
        reach_pose = grasp_pose * sapien.Pose(p=[0, 0, -0.05])
        grasp_pose = grasp_pose * sapien.Pose(p=[0, 0, 0.01])

        total_frames = defaultdict(list)
        frames_list = []
        reach_frames = self.move_to_pose(
            reach_pose,
            self.sim_steps_per_control,
            self.control_timestep,
            cameras,
            render_keys,
            gripper_state=1,
        )
        frames_list.append(reach_frames)
        if len(reach_frames) == 0:
            logger.warning(
                f"Failed to reach the grasp pose for node `{actor.name}`, skipping grasping."
            )
            return total_frames

        if not reach_target_only:
            grasp_frames = self.move_to_pose(
                grasp_pose,
                self.sim_steps_per_control,
                self.control_timestep,
                cameras,
                render_keys,
                gripper_state=1,
            )
            frames_list.append(grasp_frames)
            close_frames = self.control_gripper(
                self.sim_steps_per_control,
                cameras,
                render_keys,
                gripper_state=-1,
            )
            frames_list.append(close_frames)
            back_frames = self.move_to_pose(
                raw_tcp_pose,
                self.sim_steps_per_control,
                self.control_timestep,
                cameras,
                render_keys,
                gripper_state=-1,
            )
            frames_list.append(back_frames)

        for frame_dict in frames_list:
            for cam_name, frames in frame_dict.items():
                total_frames[cam_name].extend(frames)

        self.agent.reset(self.agent.init_qpos)

        return total_frames
