import concurrent.futures
import dataclasses
import os
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning)

from typing import Literal

import imageio
import numpy as np
import sapien.core as sapien
import sapien.physx as physx
import torch
import tyro
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from embodied_gen.data.utils import DiffrastRender
from embodied_gen.utils.geometry import MeshInfo
from embodied_gen.utils.io_utils import (
    URDFFile,
    load_json,
    load_mesh_info,
    write_json,
)
from embodied_gen.utils.log import logger

COLORMAP = list(set(ImageColor.colormap.values()))
COLOR_PALETTE = np.array(
    [ImageColor.getrgb(c) for c in COLORMAP], dtype=np.uint8
)
MAX_NUM_WORKERS = 10
MAX_GRASPS_PER_WORKER_PROCESS = 8
SAPIEN_DEFAULT_STATIC_FRICTION = 0.3
SAPIEN_DEFAULT_DYNAMIC_FRICTION = 0.3
SAPIEN_DEFAULT_RESTITUTION = 0.1

_EVALUATOR_KEEPALIVE: list["EvalGrasps"] = []
_EVALUATOR_KEEPALIVE_LOCK = threading.Lock()

__all__ = [
    "EvalGraspsConfig",
    "GripperControlInfo",
    "SapienSceneManager",
    "EvalGrasps",
    "run_eval",
    "entrypoint",
]


@dataclass
class EvalGraspsConfig:
    urdf_paths: list[str] = field(default_factory=list)
    gripper_name: str = "franka_panda"
    gripper_urdf_path: str = (
        f"embodied_gen/scripts/affordance_annot/{gripper_name}/gripper.urdf"
    )
    output_dirs: list[str] = field(default_factory=list)
    sim_freq: int = 240
    video_fps: int = 30
    render_interval: int = 24
    image_hw: tuple[int, int] = (512, 512)
    num_cameras: int = 1
    fovy_deg: float = 45.0
    sim_backend: str = "cpu"
    ray_tracing: bool = False
    gripper_move_speed: float = 0.1
    gripper_close_speed: float = 0.01
    action_wait_seconds: float = 0.5
    lift_height: float = 0.5
    sweep_distance: float = 0.5
    num_workers: int | None = None
    max_grasps_per_worker_process: int = MAX_GRASPS_PER_WORKER_PROCESS
    gripper_drive_stiffness: float = 2500.0
    gripper_static_friction: float = SAPIEN_DEFAULT_STATIC_FRICTION
    gripper_dynamic_friction: float = SAPIEN_DEFAULT_DYNAMIC_FRICTION
    gripper_restitution: float = SAPIEN_DEFAULT_RESTITUTION
    root_drive_stiffness: float = 1e6
    root_drive_damping: float = 1e4
    object_bottom_clearance: float = 0.1
    hold_object_during_close: bool = True
    slip_translation_threshold: float = 0.05
    slip_rotation_threshold_deg: float = 30.0
    check_close_slip: bool = True
    debug_mode: bool = False
    save_failed_grasps: bool = False


@dataclass
class GripperControlInfo:
    open_qpos: float
    close_qpos: float
    drive_damping: float
    force_limit: float


class GripperActionAgent:
    def __init__(
        self,
        scene_manager: "SapienSceneManager",
        gripper: physx.PhysxArticulation,
        root_target: sapien.Entity,
        root_drive: physx.PhysxDriveComponent,
    ) -> None:
        self.scene_manager = scene_manager
        self.gripper = gripper
        self.root_target = root_target
        self.root_drive = root_drive

    def set_action(self, action: np.ndarray | torch.Tensor) -> None:
        if isinstance(action, torch.Tensor):
            values = action.detach().cpu().numpy().reshape(-1)
        else:
            values = np.asarray(action).reshape(-1)
        if values.size != 8:
            raise ValueError(
                f"Gripper action must have 8 values, got {values.size}."
            )
        self.root_target.set_pose(
            sapien.Pose(
                p=values[:3].astype(np.float64),
                q=values[3:7].astype(np.float64),
            )
        )
        self.scene_manager.set_gripper_target(
            self.gripper,
            float(values[7]),
        )


class SapienSceneManager:
    def __init__(
        self, sim_freq: int, ray_tracing: bool, device: str = "cpu"
    ) -> None:
        self.sim_freq = sim_freq
        self.ray_tracing = ray_tracing
        self.device = device
        self.renderer = sapien.SapienRenderer()
        self.scene = self.setup_scene()
        self.cameras: list[sapien.render.RenderCameraComponent] = []
        self.actors: dict[str, sapien.pysapien.Entity] = {}

    def setup_scene(self) -> sapien.Scene:
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

    def set_ground_color(self, rgba: list[float]) -> None:
        """Update the default ground plane material color for this scene."""
        for actor in self.scene.get_all_actors():
            if actor.name != "ground":
                continue
            for component in actor.components:
                render_shapes = getattr(component, "render_shapes", None)
                if render_shapes is None:
                    continue
                for render_shape in render_shapes:
                    render_shape.material.set_base_color(rgba)
            return
        raise ValueError("Ground actor not found in the scene.")

    def create_camera(
        self,
        cam_name: str,
        pose: sapien.Pose,
        image_hw: tuple[int, int],
        fovy_deg: float,
    ) -> sapien.render.RenderCameraComponent:
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

    def add_ring_cameras(
        self,
        num_cameras: int,
        radius: float,
        height: float,
        target_pt: list[float],
        image_hw: tuple[int, int],
        fovy_deg: float,
    ) -> list[sapien.render.RenderCameraComponent]:
        angle_step = 2 * np.pi / num_cameras
        world_up_vec = np.array([0.0, 0.0, 1.0])
        target_pt = np.array(target_pt)

        for i in range(num_cameras):
            angle = i * angle_step
            cam_x = target_pt[0] + radius * np.cos(angle)
            cam_y = target_pt[1] + radius * np.sin(angle)
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

    def add_actor(
        self,
        mesh_info: MeshInfo,
        position: list[float],
    ) -> sapien.Entity:
        builder = self.scene.create_actor_builder()
        material = physx.PhysxMaterial(
            static_friction=float(
                np.clip(mesh_info.static_friction, 0.1, 0.7)
            ),
            dynamic_friction=float(
                np.clip(mesh_info.dynamic_friction, 0.1, 0.6)
            ),
            restitution=0.05,
        )
        builder.set_physx_body_type("dynamic")
        builder.add_multiple_convex_collisions_from_file(
            mesh_info.collision_mesh_path,
            scale=mesh_info.collision_mesh_scale,
            pose=mesh_info.collision_local_pose,
            material=material,
        )
        builder.add_visual_from_file(
            mesh_info.visual_mesh_path,
            scale=mesh_info.visual_mesh_scale,
            pose=mesh_info.visual_local_pose,
        )

        builder.set_initial_pose(sapien.Pose(p=position))
        actor = builder.build(name=mesh_info.actor_name)
        if mesh_info.mass is not None and hasattr(
            actor.components[1], "set_mass"
        ):
            actor.components[1].set_mass(mesh_info.mass)
        return actor

    def add_gripper(
        self,
        gripper_urdf_path: str,
        root_pose: sapien.Pose,
        open_qpos: float,
        drive_stiffness: float,
        drive_damping: float,
        force_limit: float,
        static_friction: float,
        dynamic_friction: float,
        restitution: float,
        root_drive_stiffness: float,
        root_drive_damping: float,
    ) -> tuple[
        physx.PhysxArticulation,
        sapien.Entity,
        physx.PhysxDriveComponent,
    ]:
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        gripper = loader.load(gripper_urdf_path)
        gripper.set_root_pose(root_pose)
        self.set_articulation_contact_material(
            gripper,
            physx.PhysxMaterial(
                static_friction,
                dynamic_friction,
                restitution,
            ),
        )
        for link in gripper.get_links():
            link.set_disable_gravity(True)

        root_target = self.scene.create_actor_builder().build_kinematic(
            name="gripper_root_target"
        )
        root_target.set_pose(root_pose)
        root_drive = self.scene.create_drive(
            root_target,
            sapien.Pose(),
            gripper.get_root(),
            sapien.Pose(),
        )
        root_drive.set_limit_x(
            0.0, 0.0, root_drive_stiffness, root_drive_damping
        )
        root_drive.set_limit_y(
            0.0, 0.0, root_drive_stiffness, root_drive_damping
        )
        root_drive.set_limit_z(
            0.0, 0.0, root_drive_stiffness, root_drive_damping
        )
        root_drive.set_limit_twist(
            0.0, 0.0, root_drive_stiffness, root_drive_damping
        )
        root_drive.set_limit_cone(
            0.0, 0.0, root_drive_stiffness, root_drive_damping
        )
        root_drive.set_drive_property_x(
            root_drive_stiffness, root_drive_damping
        )
        root_drive.set_drive_property_y(
            root_drive_stiffness, root_drive_damping
        )
        root_drive.set_drive_property_z(
            root_drive_stiffness, root_drive_damping
        )
        root_drive.set_drive_property_slerp(
            root_drive_stiffness,
            root_drive_damping,
        )
        root_drive.set_inv_mass_scales(0.0, 1.0)

        qpos = []
        for joint in gripper.get_active_joints():
            joint.set_drive_properties(
                drive_stiffness,
                drive_damping,
                force_limit,
            )
            joint.set_friction(0.1)
            limit = np.asarray(joint.get_limits()).reshape(-1, 2)[0]
            target = float(np.clip(open_qpos, limit[0], limit[1]))
            joint.set_drive_target(target)
            qpos.append(target)
        if qpos:
            gripper.set_qpos(qpos)
            gripper.set_qvel(np.zeros(len(qpos), dtype=np.float32))
        return gripper, root_target, root_drive

    @staticmethod
    def set_articulation_contact_material(
        articulation: physx.PhysxArticulation,
        material: physx.PhysxMaterial,
    ) -> None:
        for link in articulation.get_links():
            for shape in link.get_collision_shapes():
                shape.set_physical_material(material)

    def set_gripper_target(
        self,
        gripper: physx.PhysxArticulation,
        finger_qpos: float,
    ) -> None:
        for joint in gripper.get_active_joints():
            limit = np.asarray(joint.get_limits()).reshape(-1, 2)[0]
            target = float(np.clip(finger_qpos, limit[0], limit[1]))
            joint.set_drive_target(target)

    def render_images(
        self,
        camera,
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

    def step_action(
        self,
        agent: GripperActionAgent | None,
        action: np.ndarray | torch.Tensor | None,
        render_keys: list[str],
        sim_steps_per_control: int = 1,
        render: bool = True,
    ) -> dict:
        if action is not None:
            agent.set_action(action)
        frames = defaultdict(defaultdict)
        for _ in range(sim_steps_per_control):
            self.scene.step()
        if not render:
            return frames
        self.scene.update_render()
        for camera in self.cameras:
            camera.take_picture()
            images = self.render_images(camera, render_keys=render_keys)
            frames[camera.name].update(images)
        return frames


class EvalGrasps:
    def __init__(
        self,
        cfg: EvalGraspsConfig,
        mesh_info: MeshInfo,
        gripper_control: GripperControlInfo,
        grasp: dict,
    ) -> None:
        self.cfg = cfg
        self.mesh_info = mesh_info
        self.gripper_control = gripper_control
        self.grasp = grasp
        self.init_scene()
        if self.cfg.debug_mode:
            self.setup_cameras(mesh_info)
        self.object_actor = self.add_object(mesh_info)

        self.frames = {}
        self.step_index = 0
        self.world_x_offset = np.array(
            [self.cfg.sweep_distance, 0.0, 0.0],
            dtype=np.float64,
        )
        self.world_y_offset = np.array(
            [0.0, self.cfg.sweep_distance, 0.0],
            dtype=np.float64,
        )

    def should_render(self) -> bool:
        return (
            self.cfg.debug_mode
            and self.step_index % max(1, self.cfg.render_interval) == 0
        )

    def init_scene(self) -> None:
        self.scene_manager = SapienSceneManager(
            sim_freq=self.cfg.sim_freq,
            ray_tracing=self.cfg.ray_tracing,
            device=self.cfg.sim_backend,
        )
        if self.cfg.debug_mode:
            self.scene_manager.set_ground_color([0.78, 0.90, 0.72, 1.0])

    def setup_cameras(self, mesh_info: MeshInfo) -> None:
        bounds = mesh_info.transformed_mesh.bounds.copy()

        mesh_size = bounds[1] - bounds[0]
        mesh_height = max(float(mesh_size[2]), 0.5)
        target_pt = [
            float((bounds[0, 0] + bounds[1, 0]) * 0.5),
            float((bounds[0, 1] + bounds[1, 1]) * 0.5),
            float(bounds[1, 2]),
        ]

        aspect = self.cfg.image_hw[1] / self.cfg.image_hw[0]
        half_vfov = np.deg2rad(self.cfg.fovy_deg) * 0.5
        half_hfov = np.arctan(np.tan(half_vfov) * aspect)
        half_min_fov = min(half_vfov, half_hfov)
        framing_half_extents = np.array(
            [
                max(float(mesh_size[0]) * 0.5, 1e-6),
                max(float(mesh_size[1]) * 0.5, 1e-6),
                mesh_height,
            ],
            dtype=np.float64,
        )
        camera_radius = max(
            float(
                np.linalg.norm(framing_half_extents)
                / np.sin(half_min_fov)
                * 1.5
            ),
            2.5,
        )
        camera_height = target_pt[2]

        self.scene_manager.add_ring_cameras(
            num_cameras=self.cfg.num_cameras,
            radius=camera_radius,
            height=camera_height,
            target_pt=target_pt,
            image_hw=self.cfg.image_hw,
            fovy_deg=self.cfg.fovy_deg,
        )

    def add_object(
        self,
        mesh_info: MeshInfo,
    ) -> sapien.Entity:
        spawn_z = self.cfg.object_bottom_clearance - float(
            mesh_info.transformed_mesh.bounds[0, 2]
        )
        actor = self.scene_manager.add_actor(mesh_info, [0.0, 0.0, spawn_z])
        self.set_object_gravity(not self.cfg.hold_object_during_close, actor)
        return actor

    def set_object_gravity(
        self,
        enabled: bool,
        actor: sapien.Entity | None = None,
    ) -> None:
        actor = self.object_actor if actor is None else actor
        for component in actor.components:
            set_disable_gravity = getattr(
                component,
                "set_disable_gravity",
                None,
            )
            if set_disable_gravity is not None:
                set_disable_gravity(not enabled)
            if enabled:
                wake_up = getattr(component, "wake_up", None)
                if wake_up is not None:
                    wake_up()

    def run_actions(
        self,
        agent: GripperActionAgent,
        actions: np.ndarray,
    ) -> None:
        for action in actions:
            this_frames = self.scene_manager.step_action(
                agent=agent,
                action=action,
                render_keys=["Color"],
                sim_steps_per_control=1,
                render=self.should_render(),
            )
            self.update_frames(this_frames)
            self.step_index += 1

    @staticmethod
    def make_pose(
        position: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> sapien.Pose:
        return sapien.Pose(
            p=[float(v) for v in position],
            q=[float(v) for v in quat_wxyz],
        )

    @staticmethod
    def pose_matrix(pose: sapien.Pose) -> np.ndarray:
        quat_wxyz = np.asarray(pose.q, dtype=np.float64)
        quat_norm = np.linalg.norm(quat_wxyz)
        if quat_norm <= 0.0:
            raise ValueError("Pose quaternion has zero norm.")
        quat_wxyz = quat_wxyz / quat_norm

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = R.from_quat(
            [
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                quat_wxyz[0],
            ]
        ).as_matrix()
        matrix[:3, 3] = np.asarray(pose.p, dtype=np.float64)
        return matrix

    @staticmethod
    def pose_delta(
        reference_relative_pose: np.ndarray,
        current_relative_pose: np.ndarray,
    ) -> dict[str, float]:
        delta_pose = (
            np.linalg.inv(reference_relative_pose) @ current_relative_pose
        )
        translation_m = float(np.linalg.norm(delta_pose[:3, 3]))
        rotation_rad = float(R.from_matrix(delta_pose[:3, :3]).magnitude())
        return {
            "translation_m": translation_m,
            "rotation_rad": rotation_rad,
            "rotation_deg": float(np.rad2deg(rotation_rad)),
        }

    def object_rel_pose(
        self,
        agent: GripperActionAgent,
    ) -> np.ndarray:
        gripper_pose = self.pose_matrix(agent.gripper.get_root_pose())
        object_pose = self.pose_matrix(self.object_actor.get_pose())
        return np.linalg.inv(gripper_pose) @ object_pose

    def object_bottom_z(self) -> float:
        object_pose = self.pose_matrix(self.object_actor.get_pose())
        vertices = np.asarray(self.mesh_info.transformed_mesh.vertices)
        world_vertices = (object_pose[:3, :3] @ vertices.T).T + object_pose[
            :3, 3
        ]
        return float(np.min(world_vertices[:, 2]))

    def check_slip(
        self,
        agent: GripperActionAgent,
        reference_relative_pose: np.ndarray,
    ) -> dict[str, float | bool]:
        delta = self.pose_delta(
            reference_relative_pose,
            self.object_rel_pose(agent),
        )
        slipped = (
            delta["translation_m"] > self.cfg.slip_translation_threshold
            or delta["rotation_deg"] > self.cfg.slip_rotation_threshold_deg
        )
        return {
            **delta,
            "slipped": slipped,
        }

    def checked_action(
        self,
        step_name: str,
        agent: GripperActionAgent,
        action_fn,
        *,
        check_slip: bool = True,
    ) -> dict:
        reference_relative_pose = self.object_rel_pose(agent)
        action_fn()
        self.wait_scene(self.cfg.action_wait_seconds)
        delta = self.check_slip(
            agent,
            reference_relative_pose,
        )
        slipped = bool(delta["slipped"])
        success = not slipped if check_slip else True
        return {
            "step": step_name,
            "success": success,
            "checked": check_slip,
            "slipped": slipped,
            "reason": None if success else "relative_pose_slip",
            "relative_pose_delta": {
                "translation_m": delta["translation_m"],
                "rotation_rad": delta["rotation_rad"],
                "rotation_deg": delta["rotation_deg"],
            },
        }

    def get_steps(
        self,
        distance: float,
        speed: float,
    ) -> int:
        if distance <= 0.0:
            return 2
        duration = distance / speed
        return max(2, int(np.ceil(duration * self.cfg.sim_freq)))

    def move_to_pose(
        self,
        agent: GripperActionAgent,
        start_pose: sapien.Pose,
        end_pose: sapien.Pose,
        finger_qpos: float | None = None,
    ) -> None:
        if finger_qpos is None:
            finger_qpos = self.gripper_control.close_qpos

        start_pos = np.asarray(start_pose.p, dtype=np.float64)
        end_pos = np.asarray(end_pose.p, dtype=np.float64)
        travel_distance = float(np.linalg.norm(end_pos - start_pos))
        num_steps = self.get_steps(
            travel_distance,
            self.cfg.gripper_move_speed,
        )
        alphas = np.linspace(0.0, 1.0, num_steps)
        quat = np.asarray(start_pose.q, dtype=np.float64)

        actions = []
        for alpha in alphas:
            pose = self.make_pose(
                (1.0 - alpha) * start_pos + alpha * end_pos,
                quat,
            )
            actions.append(self.make_action(pose, finger_qpos))
        self.run_actions(agent, np.stack(actions, axis=0))

    def move_gripper(
        self,
        agent: GripperActionAgent,
        offset: np.ndarray,
    ) -> None:
        start_pose = agent.gripper.get_root_pose()
        end_pose = self.make_pose(
            np.asarray(start_pose.p, dtype=np.float64)
            + np.asarray(offset, dtype=np.float64),
            np.asarray(start_pose.q, dtype=np.float64),
        )
        self.move_to_pose(agent, start_pose, end_pose)

    def init_grasp_pose(self) -> sapien.Pose:
        orientation = self.grasp["orientation"]
        self.grasp = sapien.Pose(
            p=[float(v) for v in self.grasp["position"]],
            q=[
                float(orientation["w"]),
                float(orientation["xyz"][0]),
                float(orientation["xyz"][1]),
                float(orientation["xyz"][2]),
            ],
        )
        gripper_pose = self.object_actor.get_pose() * self.grasp
        return gripper_pose

    def make_action(
        self,
        pose: sapien.Pose,
        finger_qpos: float,
    ) -> np.ndarray:
        return np.array(
            [
                float(pose.p[0]),
                float(pose.p[1]),
                float(pose.p[2]),
                float(pose.q[0]),
                float(pose.q[1]),
                float(pose.q[2]),
                float(pose.q[3]),
                float(finger_qpos),
            ],
            dtype=np.float32,
        )

    def add_gripper_agent(self) -> GripperActionAgent:
        gripper_pose = self.init_grasp_pose()
        gripper, root_target, root_drive = self.scene_manager.add_gripper(
            self.cfg.gripper_urdf_path,
            gripper_pose,
            self.gripper_control.open_qpos,
            self.cfg.gripper_drive_stiffness,
            self.gripper_control.drive_damping,
            self.gripper_control.force_limit,
            self.cfg.gripper_static_friction,
            self.cfg.gripper_dynamic_friction,
            self.cfg.gripper_restitution,
            self.cfg.root_drive_stiffness,
            self.cfg.root_drive_damping,
        )
        return GripperActionAgent(
            self.scene_manager,
            gripper,
            root_target,
            root_drive,
        )

    def update_frames(self, this_frames: dict) -> None:
        for cam_name, images_list in this_frames.items():
            if cam_name not in self.frames:
                self.frames[cam_name] = []
            self.frames[cam_name].append(np.array(images_list["Color"]))

    def wait_scene(self, seconds: float) -> None:
        num_steps = int(self.cfg.sim_freq * seconds)
        for _ in range(num_steps):
            this_frames = self.scene_manager.step_action(
                agent=None,
                action=None,
                render_keys=["Color"],
                sim_steps_per_control=1,
                render=self.should_render(),
            )
            self.update_frames(this_frames)
            self.step_index += 1

    def close_gripper(self, agent: GripperActionAgent) -> None:
        num_close_steps = self.get_steps(
            abs(
                self.gripper_control.open_qpos
                - self.gripper_control.close_qpos
            ),
            self.cfg.gripper_close_speed,
        )
        root_pose = agent.gripper.get_root_pose()
        actions = [
            self.make_action(
                root_pose,
                (1.0 - alpha) * self.gripper_control.open_qpos
                + alpha * self.gripper_control.close_qpos,
            )
            for alpha in np.linspace(0.0, 1.0, max(1, num_close_steps))
        ]
        self.run_actions(agent, np.stack(actions, axis=0))

    @staticmethod
    def summarize_steps(step_results: list[dict]) -> dict:
        failure = next(
            (
                step_result
                for step_result in step_results
                if not step_result["success"]
            ),
            None,
        )
        result = {
            "success": failure is None,
            "relative_pose_deltas": {
                step_result["step"]: step_result["relative_pose_delta"]
                for step_result in step_results
            },
        }
        if failure is not None:
            result.update(
                {
                    "failure_step": failure["step"],
                    "reason": failure["reason"],
                }
            )
        return result

    def run(self) -> dict:
        step_results = []
        self.wait_scene(2.0)

        this_agent = self.add_gripper_agent()
        self.wait_scene(self.cfg.action_wait_seconds)

        # Close the gripper around the object.
        step_result = self.checked_action(
            "close_gripper",
            this_agent,
            lambda: self.close_gripper(this_agent),
            check_slip=self.cfg.check_close_slip,
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        if self.cfg.hold_object_during_close:
            self.set_object_gravity(True)

        # Lift the grasped object upward.
        step_result = self.checked_action(
            "lift_gripper",
            this_agent,
            lambda: self.move_gripper(
                this_agent,
                np.array(
                    [0.0, 0.0, self.cfg.lift_height],
                    dtype=np.float64,
                ),
            ),
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        shake_start_pose = this_agent.gripper.get_root_pose()
        shake_end_pose = self.make_pose(
            np.asarray(shake_start_pose.p, dtype=np.float64)
            + self.world_x_offset,
            np.asarray(shake_start_pose.q, dtype=np.float64),
        )

        # Move the gripper outward along the X axis.
        step_result = self.checked_action(
            "move_gripper_out_x",
            this_agent,
            lambda: self.move_to_pose(
                this_agent,
                shake_start_pose,
                shake_end_pose,
            ),
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        # Move the gripper back from the X-axis offset.
        step_result = self.checked_action(
            "move_gripper_back_x",
            this_agent,
            lambda: self.move_to_pose(
                this_agent,
                this_agent.gripper.get_root_pose(),
                shake_start_pose,
            ),
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        shake_start_pose = this_agent.gripper.get_root_pose()
        shake_end_pose = self.make_pose(
            np.asarray(shake_start_pose.p, dtype=np.float64)
            + self.world_y_offset,
            np.asarray(shake_start_pose.q, dtype=np.float64),
        )

        # Move the gripper outward along the Y axis.
        step_result = self.checked_action(
            "move_gripper_out_y",
            this_agent,
            lambda: self.move_to_pose(
                this_agent,
                shake_start_pose,
                shake_end_pose,
            ),
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        # Move the gripper back from the Y-axis offset.
        step_result = self.checked_action(
            "move_gripper_back_y",
            this_agent,
            lambda: self.move_to_pose(
                this_agent,
                this_agent.gripper.get_root_pose(),
                shake_start_pose,
            ),
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        # Lower the grasped object back down.
        descend_offset = np.array(
            [0.0, 0.0, 0.01 - self.object_bottom_z()],
            dtype=np.float64,
        )
        step_result = self.checked_action(
            "descend_gripper",
            this_agent,
            lambda: self.move_gripper(this_agent, descend_offset),
        )
        step_results.append(step_result)
        if not step_result["success"]:
            return self.summarize_steps(step_results)

        return self.summarize_steps(step_results)

    def export_video(self, output_dir: str) -> None:
        if not self.cfg.debug_mode:
            return
        os.makedirs(output_dir, exist_ok=True)
        for cam_name, this_frames in self.frames.items():
            this_output_path = os.path.join(output_dir, f"{cam_name}.mp4")
            imageio.mimsave(
                this_output_path, this_frames, fps=self.cfg.video_fps
            )


def _validate_config(cfg: EvalGraspsConfig) -> None:
    if not cfg.urdf_paths:
        raise ValueError("urdf_paths must be provided.")

    if len(cfg.output_dirs) == 0:
        cfg.output_dirs = [
            os.path.join(os.path.dirname(path), "affordance")
            for path in cfg.urdf_paths
        ]

    if len(cfg.urdf_paths) != len(cfg.output_dirs):
        raise ValueError(
            "urdf_paths and output_dirs must have the same length, "
            f"got {len(cfg.urdf_paths)} and {len(cfg.output_dirs)}."
        )


def _set_single_config(
    cfg: EvalGraspsConfig,
    urdf_path: str,
    output_dir: str,
) -> EvalGraspsConfig:
    return dataclasses.replace(
        cfg,
        urdf_paths=[urdf_path],
        output_dirs=[output_dir],
    )


def _eval_grasp(
    cfg: EvalGraspsConfig,
    grasp_id: str,
    grasp: dict,
) -> dict:
    output_dir = cfg.output_dirs[0]

    mesh_info = load_mesh_info(cfg.urdf_paths[0])
    gripper_control = GripperControlInfo(
        **URDFFile(cfg.gripper_urdf_path).get_prismatic_joint_control_info()
    )
    evaluator = EvalGrasps(cfg, mesh_info, gripper_control, grasp)
    result = evaluator.run()
    evaluator.export_video(
        os.path.join(output_dir, "eval_grasp_renders", grasp_id)
    )
    if threading.current_thread() is not threading.main_thread():
        with _EVALUATOR_KEEPALIVE_LOCK:
            _EVALUATOR_KEEPALIVE.append(evaluator)
    return {
        "grasp_id": grasp_id,
        "output_dir": output_dir,
        "simulation_result": result,
    }


def _release_thread_evaluators() -> None:
    with _EVALUATOR_KEEPALIVE_LOCK:
        evaluators = list(_EVALUATOR_KEEPALIVE)
        _EVALUATOR_KEEPALIVE.clear()
    for evaluator in evaluators:
        del evaluator


def _eval_grasp_task(task: tuple[EvalGraspsConfig, str, dict]) -> dict:
    cfg, grasp_id, grasp = task
    return _eval_grasp(cfg, grasp_id, grasp)


def _slip_thresholds(cfg: EvalGraspsConfig) -> dict[str, float]:
    return {
        "translation_m": cfg.slip_translation_threshold,
        "rotation_deg": cfg.slip_rotation_threshold_deg,
    }


def _is_successful_grasp_result(result: dict) -> bool:
    simulation_result = result.get("simulation_result", {})
    if not isinstance(simulation_result, dict):
        return False
    return bool(simulation_result.get("success", False))


def _grasp_success_by_id(results: list[dict]) -> dict[str, bool]:
    return {
        result["grasp_id"]: _is_successful_grasp_result(result)
        for result in results
    }


def _build_eval_summary(
    cfg: EvalGraspsConfig,
    results: list[dict],
    elapsed_seconds: float,
) -> dict:
    successful_grasp_ids = [
        result["grasp_id"]
        for result in results
        if _is_successful_grasp_result(result)
    ]
    failed_grasp_ids = [
        result["grasp_id"]
        for result in results
        if not _is_successful_grasp_result(result)
    ]
    num_grasps = len(results)
    num_success = len(successful_grasp_ids)
    return {
        "num_grasps": num_grasps,
        "num_success": num_success,
        "num_failure": len(failed_grasp_ids),
        "success_rate": num_success / max(1, num_grasps),
        "successful_grasp_ids": successful_grasp_ids,
        "failed_grasp_ids": failed_grasp_ids,
        "slip_thresholds": _slip_thresholds(cfg),
        "elapsed_seconds": elapsed_seconds,
    }


def _run_single_eval(cfg: EvalGraspsConfig) -> None:
    start_time = time.perf_counter()
    output_dir = cfg.output_dirs[0]
    os.makedirs(output_dir, exist_ok=True)
    urdf_path = cfg.urdf_paths[0]
    urdf = URDFFile(urdf_path)

    affordance_path = urdf.get_affordance_annot_path()
    affordance_payload = load_json(affordance_path)
    affordances = affordance_payload.get("affordances", [])
    grasps = {}
    for affordance in affordances:
        grasps.update(affordance.get("grasp_group", {}))

    if not grasps:
        return

    num_workers = cfg.num_workers
    if num_workers is None:
        num_workers = min(len(grasps), os.cpu_count() or 1, MAX_NUM_WORKERS)
    else:
        num_workers = min(num_workers, len(grasps), MAX_NUM_WORKERS)

    logger.info(
        f"Running {len(grasps)} grasp poses with {num_workers} worker(s)"
    )
    results = []
    if num_workers == 1:
        for grasp_id, grasp in tqdm(grasps.items(), desc="Evaluating grasps"):
            results.append(_eval_grasp(cfg, grasp_id, grasp))
    else:
        worker_cfg = dataclasses.replace(cfg, num_workers=1)
        tasks = [
            (worker_cfg, grasp_id, grasp) for grasp_id, grasp in grasps.items()
        ]
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = [
                    executor.submit(_eval_grasp_task, task) for task in tasks
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Evaluating grasps",
                ):
                    results.append(future.result())
            results.sort(
                key=lambda result: list(grasps.keys()).index(
                    result["grasp_id"]
                )
            )
        finally:
            _release_thread_evaluators()

    elapsed_seconds = time.perf_counter() - start_time
    summary = _build_eval_summary(cfg, results, elapsed_seconds)
    failed_grasp_ids = set(summary["failed_grasp_ids"])
    grasp_success_by_id = _grasp_success_by_id(results)
    for affordance in affordance_payload.get("affordances", []):
        grasp_group = affordance.get("grasp_group")
        if isinstance(grasp_group, dict):
            if cfg.debug_mode or cfg.save_failed_grasps:
                for grasp_id, grasp in grasp_group.items():
                    if (
                        isinstance(grasp, dict)
                        and grasp_id in grasp_success_by_id
                    ):
                        grasp["simulation_success"] = grasp_success_by_id[
                            grasp_id
                        ]
                continue
            for failed_grasp_id in failed_grasp_ids:
                grasp_group.pop(failed_grasp_id, None)

    write_json(affordance_payload, affordance_path)

    logger.info(f"Grasp evaluation simulation took {elapsed_seconds:.2f}s")
    logger.info(f"Updated grasp simulation results in {affordance_path}")
    logger.info(
        f"Grasp success rate: {summary['num_success']}/"
        f"{summary['num_grasps']} ({summary['success_rate']:.2%})"
    )


def _run_single_eval_safe(cfg: EvalGraspsConfig) -> bool:
    urdf_path = cfg.urdf_paths[0]
    try:
        _run_single_eval(cfg)
    except Exception as exc:
        logger.error(
            "Grasp evaluation failed for URDF {}: {}".format(
                urdf_path,
                exc,
            )
        )
        return False
    return True


def run_eval(cfg: EvalGraspsConfig) -> None:
    _validate_config(cfg)
    for idx, (urdf_path, output_dir) in enumerate(
        zip(cfg.urdf_paths, cfg.output_dirs),
        start=1,
    ):
        single_cfg = _set_single_config(cfg, urdf_path, output_dir)
        logger.info(
            f"Running grasp evaluation for URDF {idx}/{len(cfg.urdf_paths)}: "
            f"{urdf_path}"
        )
        _run_single_eval_safe(single_cfg)


def entrypoint(*args, **kwargs) -> None:
    cfg = tyro.cli(EvalGraspsConfig)
    run_eval(cfg)


if __name__ == "__main__":
    entrypoint()
