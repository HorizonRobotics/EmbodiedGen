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
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Literal

import mplib
import numpy as np
import sapien.core as sapien
import sapien.physx as physx
import torch
import trimesh
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
)
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R
from embodied_gen.data.utils import DiffrastRender
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.geometry import quaternion_multiply
from embodied_gen.utils.log import logger

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
    "load_assets_from_layout_file",
    "load_mani_skill_robot",
    "render_images",
    "is_urdf_articulated",
]


def is_urdf_articulated(urdf_path: str) -> bool:
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            j_type = joint.get("type")
            if j_type in ["prismatic", "revolute", "continuous", "planar"]:
                return True
        return False
    except Exception as e:
        print(f"Error parsing URDF {urdf_path}: {e}.")
        return False


def load_actor_from_urdf(
    scene: sapien.Scene | ManiSkillScene,
    file_path: str,
    pose: sapien.Pose | None = None,
    env_idx: int = None,
    use_static: bool = False,
    update_mass: bool = False,
    scale: float | np.ndarray = 1.0,
) -> sapien.pysapien.Entity:
    """Load an sapien actor from a URDF file and add it to the scene.

    Args:
        scene (sapien.Scene | ManiSkillScene): The simulation scene.
        file_path (str): Path to the URDF file.
        pose (sapien.Pose | None): Initial pose of the actor.
        env_idx (int): Environment index for multi-env setup.
        use_static (bool): Whether the actor is static.
        update_mass (bool): Whether to update the actor's mass from URDF.
        scale (float | np.ndarray): Scale factor for the actor.

    Returns:
        sapien.pysapien.Entity: The created actor entity.
    """

    def _get_local_pose(origin_tag: ET.Element | None) -> sapien.Pose:
        local_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if origin_tag is not None:
            xyz = list(map(float, origin_tag.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin_tag.get("rpy", "0 0 0").split()))
            qx, qy, qz, qw = R.from_euler("xyz", rpy, degrees=False).as_quat()
            local_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])

        return local_pose

    tree = ET.parse(file_path)
    root = tree.getroot()
    node_name = root.get("name")
    file_dir = os.path.dirname(file_path)

    visual_mesh = root.find(".//visual/geometry/mesh")
    visual_file = visual_mesh.get("filename")
    visual_scale = visual_mesh.get("scale", "1.0 1.0 1.0")
    visual_scale = np.array([float(x) for x in visual_scale.split()])
    visual_scale *= np.array(scale)

    collision_mesh = root.find(".//collision/geometry/mesh")
    collision_file = collision_mesh.get("filename")
    collision_scale = collision_mesh.get("scale", "1.0 1.0 1.0")
    collision_scale = np.array([float(x) for x in collision_scale.split()])
    collision_scale *= np.array(scale)

    visual_pose = _get_local_pose(root.find(".//visual/origin"))
    collision_pose = _get_local_pose(root.find(".//collision/origin"))

    visual_file = os.path.join(file_dir, visual_file)
    collision_file = os.path.join(file_dir, collision_file)
    static_fric = root.find(".//collision/gazebo/mu1").text
    dynamic_fric = root.find(".//collision/gazebo/mu2").text

    material = physx.PhysxMaterial(
        static_friction=np.clip(float(static_fric), 0.1, 0.7),
        dynamic_friction=np.clip(float(dynamic_fric), 0.1, 0.6),
        restitution=0.05,
    )
    builder = scene.create_actor_builder()

    body_type = "static" if use_static else "dynamic"
    builder.set_physx_body_type(body_type)
    builder.add_multiple_convex_collisions_from_file(
        collision_file,
        material=material,
        scale=collision_scale,
        # decomposition="coacd",
        # decomposition_params=dict(
        #     threshold=0.05, max_convex_hull=64, verbose=False
        # ),
        pose=collision_pose,
    )

    builder.add_visual_from_file(
        visual_file,
        scale=visual_scale,
        pose=visual_pose,
    )
    if pose is None:
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])

    builder.set_initial_pose(pose)
    if isinstance(scene, ManiSkillScene) and env_idx is not None:
        builder.set_scene_idxs([env_idx])

    actor = builder.build(
        name=node_name if env_idx is None else f"{node_name}-{env_idx}"
    )

    if update_mass and hasattr(actor.components[1], "mass"):
        node_mass = float(root.find(".//inertial/mass").get("value"))
        actor.components[1].set_mass(node_mass)

    return actor


def load_assets_from_layout_file(
    scene: ManiSkillScene | sapien.Scene,
    layout: str,
    z_offset: float = 0.0,
    init_quat: list[float] = [0, 0, 0, 1],
    env_idx: int = None,
) -> dict[str, sapien.pysapien.Entity]:
    """Load assets from an EmbodiedGen layout file and create sapien actors in the scene.

    Args:
        scene (ManiSkillScene | sapien.Scene): The sapien simulation scene.
        layout (str): Path to the embodiedgen layout file.
        z_offset (float): Z offset for non-context objects.
        init_quat (list[float]): Initial quaternion for orientation.
        env_idx (int): Environment index.

    Returns:
        dict[str, sapien.pysapien.Entity]: Mapping from object names to actor entities.
    """
    asset_root = os.path.dirname(layout)
    layout = LayoutInfo.from_dict(json.load(open(layout, "r")))
    actors = dict()
    for node in layout.assets:
        file_dir = layout.assets[node]
        file_name = f"{node.replace(' ', '_')}.urdf"
        urdf_file = os.path.join(asset_root, file_dir, file_name)

        if layout.objs_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
            continue

        position = layout.position[node].copy()
        if layout.objs_mapping[node] != Scene3DItemEnum.CONTEXT.value:
            position[2] += z_offset

        use_static = (
            layout.relation.get(Scene3DItemEnum.CONTEXT.value, None) == node
        )

        # Combine initial quaternion with object quaternion
        x, y, z, qx, qy, qz, qw = position
        qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw], init_quat)
        target_pose = sapien.Pose(p=[x, y, z], q=[qw, qx, qy, qz])
        if is_urdf_articulated(urdf_file):
            loader = scene.create_urdf_loader()
            loader.fix_root_link = use_static
            actor = loader.load(urdf_file)
            actor.set_root_pose(target_pose)
        else:
            actor = load_actor_from_urdf(
                scene,
                urdf_file,
                target_pose,
                env_idx,
                use_static=use_static,
                update_mass=False,
            )
        actors[node] = actor

    return actors


def load_mani_skill_robot(
    scene: sapien.Scene | ManiSkillScene,
    layout: LayoutInfo | str,
    control_freq: int = 20,
    robot_init_qpos_noise: float = 0.0,
    control_mode: str = "pd_joint_pos",
    backend_str: tuple[str, str] = ("cpu", "gpu"),
) -> BaseAgent:
    """Load a ManiSkill robot agent into the scene.

    Args:
        scene (sapien.Scene | ManiSkillScene): The simulation scene.
        layout (LayoutInfo | str): Layout info or path to layout file.
        control_freq (int): Control frequency.
        robot_init_qpos_noise (float): Noise for initial joint positions.
        control_mode (str): Robot control mode.
        backend_str (tuple[str, str]): Simulation/render backend.

    Returns:
        BaseAgent: The loaded robot agent.
    """
    from mani_skill.agents import REGISTERED_AGENTS
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.envs.utils.system.backend import (
        parse_sim_and_render_backend,
    )

    if isinstance(layout, str) and layout.endswith(".json"):
        layout = LayoutInfo.from_dict(json.load(open(layout, "r")))

    robot_name = layout.relation[Scene3DItemEnum.ROBOT.value]
    x, y, z, qx, qy, qz, qw = layout.position[robot_name]
    delta_z = 0.002  # Add small offset to avoid collision.
    pose = sapien.Pose([x, y, z + delta_z], [qw, qx, qy, qz])

    if robot_name not in REGISTERED_AGENTS:
        logger.warning(
            f"Robot `{robot_name}` not registered, chosen from {REGISTERED_AGENTS.keys()}, use `panda` instead."
        )
        robot_name = "panda"

    ROBOT_CLS = REGISTERED_AGENTS[robot_name].agent_cls
    backend = parse_sim_and_render_backend(*backend_str)
    if isinstance(scene, sapien.Scene):
        scene = ManiSkillScene([scene], device=backend_str[0], backend=backend)
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


def render_images(
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
    """Render images from a given SAPIEN camera.

    Args:
        camera (sapien.render.RenderCameraComponent): Camera to render from.
        render_keys (list[str], optional): Types of images to render.

    Returns:
        dict[str, Image.Image]: Dictionary of rendered images.
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
        seg_color = COLOR_PALETTE[label0]
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


class SapienSceneManager:
    """Manages SAPIEN simulation scenes, cameras, and rendering.

    This class provides utilities for setting up scenes, adding cameras,
    stepping simulation, and rendering images.

    Attributes:
        sim_freq (int): Simulation frequency.
        ray_tracing (bool): Whether to use ray tracing.
        device (str): Device for simulation.
        renderer (sapien.SapienRenderer): SAPIEN renderer.
        scene (sapien.Scene): Simulation scene.
        cameras (list): List of camera components.
        actors (dict): Mapping of actor names to entities.

    Example see `embodied_gen/scripts/simulate_sapien.py`.
    """

    def __init__(
        self, sim_freq: int, ray_tracing: bool, device: str = "cuda"
    ) -> None:
        """Initialize the scene manager.

        Args:
            sim_freq (int): Simulation frequency.
            ray_tracing (bool): Enable ray tracing.
            device (str): Device for simulation.
        """
        self.sim_freq = sim_freq
        self.ray_tracing = ray_tracing
        self.device = device
        self.renderer = sapien.SapienRenderer()
        self.scene = self._setup_scene()
        self.cameras: list[sapien.render.RenderCameraComponent] = []
        self.actors: dict[str, sapien.pysapien.Entity] = {}

    def _setup_scene(self) -> sapien.Scene:
        """Set up the SAPIEN scene with lighting and ground.

        Returns:
            sapien.Scene: The initialized scene.
        """
        # Ray tracing settings
        if self.ray_tracing:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(10)
            sapien.render.set_ray_tracing_denoiser("oidn")

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
        agent: BaseAgent,
        action: torch.Tensor,
        cameras: list[sapien.render.RenderCameraComponent],
        render_keys: list[str],
        sim_steps_per_control: int = 1,
    ) -> dict:
        """Step the simulation and render images from cameras.

        Args:
            agent (BaseAgent): The robot agent.
            action (torch.Tensor): Action to apply.
            cameras (list): List of camera components.
            render_keys (list[str]): Types of images to render.
            sim_steps_per_control (int): Simulation steps per control.

        Returns:
            dict: Dictionary of rendered frames per camera.
        """
        agent.set_action(action)
        frames = defaultdict(list)
        for _ in range(sim_steps_per_control):
            self.scene.step()

        self.scene.update_render()
        for camera in cameras:
            camera.take_picture()
            images = render_images(camera, render_keys=render_keys)
            frames[camera.name].append(images)

        return frames

    def create_camera(
        self,
        cam_name: str,
        pose: sapien.Pose,
        image_hw: tuple[int, int],
        fovy_deg: float,
    ) -> sapien.render.RenderCameraComponent:
        """Create a camera in the scene.

        Args:
            cam_name (str): Camera name.
            pose (sapien.Pose): Camera pose.
            image_hw (tuple[int, int]): Image resolution (height, width).
            fovy_deg (float): Field of view in degrees.

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
            num_cameras (int): Number of cameras.
            radius (float): Circle radius.
            height (float): Camera height.
            target_pt (list[float]): Target point to look at.
            image_hw (tuple[int, int]): Image resolution.
            fovy_deg (float): Field of view in degrees.

        Returns:
            list[sapien.render.RenderCameraComponent]: List of cameras.
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


class FrankaPandaGrasper(object):
    """Provides grasp planning and control for Franka Panda robot.

    Attributes:
        agent (BaseAgent): The robot agent.
        robot: The robot instance.
        control_freq (float): Control frequency.
        control_timestep (float): Control timestep.
        joint_vel_limits (float): Joint velocity limits.
        joint_acc_limits (float): Joint acceleration limits.
        finger_length (float): Length of gripper fingers.
        planners: Motion planners for each environment.
    """

    def __init__(
        self,
        agent: BaseAgent,
        control_freq: float,
        joint_vel_limits: float = 2.0,
        joint_acc_limits: float = 1.0,
        finger_length: float = 0.025,
    ) -> None:
        """Initialize the grasper."""
        self.agent = agent
        self.robot = agent.robot
        self.control_freq = control_freq
        self.control_timestep = 1 / control_freq
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.finger_length = finger_length
        self.planners = self._setup_planner()

    def _setup_planner(self) -> mplib.Planner:
        planners = []
        for pose in self.robot.pose:
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
            planner.set_base_pose(pose.raw_pose[0].tolist())
            planners.append(planner)

        return planners

    def control_gripper(
        self,
        gripper_state: Literal[-1, 1],
        n_step: int = 10,
    ) -> np.ndarray:
        """Generate gripper control actions.

        Args:
            gripper_state (Literal[-1, 1]): Desired gripper state.
            n_step (int): Number of steps.

        Returns:
            np.ndarray: Array of gripper actions.
        """
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        actions = []
        for _ in range(n_step):
            action = np.hstack([qpos, gripper_state])[None, ...]
            actions.append(action)

        return np.concatenate(actions, axis=0)

    def move_to_pose(
        self,
        pose: sapien.Pose,
        control_timestep: float,
        gripper_state: Literal[-1, 1],
        use_point_cloud: bool = False,
        n_max_step: int = 100,
        action_key: str = "position",
        env_idx: int = 0,
    ) -> np.ndarray:
        """Plan and execute motion to a target pose.

        Args:
            pose (sapien.Pose): Target pose.
            control_timestep (float): Control timestep.
            gripper_state (Literal[-1, 1]): Desired gripper state.
            use_point_cloud (bool): Use point cloud for planning.
            n_max_step (int): Max number of steps.
            action_key (str): Key for action in result.
            env_idx (int): Environment index.

        Returns:
            np.ndarray: Array of actions to reach the pose.
        """
        result = self.planners[env_idx].plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=control_timestep,
            use_point_cloud=use_point_cloud,
        )

        if result["status"] != "Success":
            result = self.planners[env_idx].plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=control_timestep,
                use_point_cloud=use_point_cloud,
            )

        if result["status"] != "Success":
            return

        sample_ratio = (len(result[action_key]) // n_max_step) + 1
        result[action_key] = result[action_key][::sample_ratio]

        n_step = len(result[action_key])
        actions = []
        for i in range(n_step):
            qpos = result[action_key][i]
            action = np.hstack([qpos, gripper_state])[None, ...]
            actions.append(action)

        return np.concatenate(actions, axis=0)

    def compute_grasp_action(
        self,
        actor: sapien.pysapien.Entity,
        reach_target_only: bool = True,
        offset: tuple[float, float, float] = [0, 0, -0.05],
        env_idx: int = 0,
    ) -> np.ndarray:
        """Compute grasp actions for a target actor.

        Args:
            actor (sapien.pysapien.Entity): Target actor to grasp.
            reach_target_only (bool): Only reach the target pose if True.
            offset (tuple[float, float, float]): Offset for reach pose.
            env_idx (int): Environment index.

        Returns:
            np.ndarray: Array of grasp actions.
        """
        if isinstance(actor, physx.PhysxArticulation):
            meshes = []
            for link in actor.links:
                link_mesh = get_component_mesh(link, to_world_frame=True)
                if link_mesh is not None and not link_mesh.is_empty:
                    meshes.append(link_mesh)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                logger.warning(
                    f"Articulation {actor.name} has no valid meshes."
                )
                return None
        else:
            physx_rigid = actor.components[1]
            mesh = get_component_mesh(physx_rigid, to_world_frame=True)

        obb = mesh.bounding_box_oriented
        approaching = np.array([0, 0, -1])
        tcp_pose = self.agent.tcp.pose[env_idx]
        target_closing = (
            tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=self.finger_length,
        )

        closing, center = grasp_info["closing"], grasp_info["center"]
        raw_tcp_pose = tcp_pose.sp
        grasp_pose = self.agent.build_grasp_pose(approaching, closing, center)
        reach_pose = grasp_pose * sapien.Pose(p=offset)
        grasp_pose = grasp_pose * sapien.Pose(p=[0, 0, 0.01])
        actions = []
        reach_actions = self.move_to_pose(
            reach_pose,
            self.control_timestep,
            gripper_state=1,
            env_idx=env_idx,
        )
        actions.append(reach_actions)

        if reach_actions is None:
            logger.warning(
                f"Failed to reach the grasp pose for node `{actor.name}`, skipping grasping."
            )
            return None

        if not reach_target_only:
            grasp_actions = self.move_to_pose(
                grasp_pose,
                self.control_timestep,
                gripper_state=1,
                env_idx=env_idx,
            )
            actions.append(grasp_actions)
            close_actions = self.control_gripper(
                gripper_state=-1,
                env_idx=env_idx,
            )
            actions.append(close_actions)
            back_actions = self.move_to_pose(
                raw_tcp_pose,
                self.control_timestep,
                gripper_state=-1,
                env_idx=env_idx,
            )
            actions.append(back_actions)

        return np.concatenate(actions, axis=0)
