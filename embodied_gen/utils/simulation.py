import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Literal, Tuple

import numpy as np
import sapien.core as sapien
import torch
from PIL import Image, ImageColor
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from embodied_gen.data.utils import DiffrastRender
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum

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
    "SapienSceneManager",
]


class SapienSceneManager:
    """A class to manage SAPIEN simulator."""

    def __init__(self, sim_freq: int):
        """Initialize the SAPIEN scene manager.

        Args:
            sim_freq (int): Simulation frequency (Hz).
        """
        self.sim_freq = sim_freq
        self.renderer = sapien.SapienRenderer()
        self.scene = self._setup_scene()
        self.cameras: List[sapien.render.RenderCameraComponent] = []
        self.actors: List[sapien.pysapien.Entity] = []

    def _setup_scene(self) -> sapien.Scene:
        """Set up the SAPIEN scene with lighting and ground."""
        # Ray tracing settings
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

        self.actors.append(actor)

        return actor

    def create_camera(
        self,
        cam_name: str,
        pose: sapien.Pose,
        image_hw: Tuple[int, int],
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
        image_hw: Tuple[int, int],
        fovy_deg: float,
    ) -> List[sapien.render.RenderCameraComponent]:
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
        render_keys: List[
            Literal[
                "Color",
                "Segmentation",
                "Normal",
                "Mask",
                "Depth",
                "Foreground",
            ]
        ] = None,
    ) -> Dict[str, Image.Image]:
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

        results: Dict[str, Image.Image] = {}
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
        init_quat: List[float] = [1, 0, 0, 0],
    ) -> None:
        """Load assets from `EmbodiedGen` layout-gen output and create actors in the scene.

        Args:
            layout (LayoutInfo): The layout information data.
            z_offset (float): Offset to apply to the Z-coordinate of non-context objects.
            init_quat (List[float]): Initial quaternion (w, x, y, z) for orientation adjustment.
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

            # Combine initial quaternion with object quaternion
            qx, qy, qz, qw = init_quat
            q1 = Quaternion(w=qw, x=qx, y=qy, z=qz)
            x, y, z, qx, qy, qz, qw = position
            q2 = Quaternion(w=qw, x=qx, y=qy, z=qz)
            quat = q2 * q1

            use_static = (
                layout.relation.get(Scene3DItemEnum.CONTEXT.value, None)
                == node
            )

            actor = self.load_actor_from_urdf(
                urdf_file,
                sapien.Pose(p=[x, y, z], q=[quat.w, quat.x, quat.y, quat.z]),
                use_static=use_static,
                update_mass=False,
            )

        return
