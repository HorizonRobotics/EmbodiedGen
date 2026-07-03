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


from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bpy
import cv2
import numpy as np
from mathutils import Euler, Matrix, Vector

logger = logging.getLogger(__name__)

DOOR_KEYWORDS = ("door",)
"""Object-name keywords used to identify door meshes for removal."""

CAMERA_TRAJECTORY_CSV_FIELDS = (
    "frame",
    "time_sec",
    "x",
    "y",
    "z",
    "roll_x_deg",
    "pitch_y_deg",
    "yaw_z_deg",
)
"""Required fields for a camera trajectory CSV."""


@dataclass(frozen=True)
class CameraTrajectoryFrame:
    """One fully specified camera pose from a trajectory CSV."""

    frame_index: int
    time_sec: float
    xyz: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for USD rendering."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd_path", required=True, type=Path)
    parser.add_argument("--glb_path", type=str, default="")
    parser.add_argument(
        "--glb_xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--glb_rotation_deg",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--render_passes",
        nargs="+",
        choices=("rgb", "depth", "normal", "mesh", "instance_seg", "flow"),
        default=("rgb",),
    )
    parser.add_argument(
        "--depth_mode",
        choices=("normalized", "metric"),
        default="normalized",
        help=(
            "normalized: per-frame min-max (default); metric: fixed "
            "0..--depth_max mapping, consistent across frames."
        ),
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=10.0,
        help="Max depth (m) mapped to white in metric depth mode.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument(
        "--camera_xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Required for single-frame mode; ignored with --trajectory_json "
            "or --trajectory_csv."
        ),
    )
    parser.add_argument(
        "--camera_rotation_deg",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        help=(
            "Required for single-frame mode; ignored with --trajectory_json "
            "or --trajectory_csv."
        ),
    )
    parser.add_argument(
        "--flow_camera_xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--flow_camera_rotation_deg",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
    )
    parser.add_argument("--focal_length_mm", type=float, default=20.0)
    parser.add_argument("--exposure", type=float, default=2.2)
    parser.add_argument("--world_strength", type=float, default=8.0)
    parser.add_argument("--fill_light_energy", type=float, default=14000.0)
    parser.add_argument(
        "--remove_doors",
        action="store_true",
        help="Remove all door meshes from the USD before rendering.",
    )

    # Trajectory modes: roaming JSON follows (x, y, rot) waypoints with a GLB
    # offset, while camera CSV uses a complete 6DoF camera pose per row.
    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--trajectory_json",
        type=Path,
        default=None,
        help=(
            "Trajectory JSON ({waypoints:[{x,y,rot,...}]}). When set, renders "
            "one frame per waypoint instead of a single frame."
        ),
    )
    trajectory_group.add_argument(
        "--trajectory_csv",
        type=Path,
        default=None,
        help=(
            "Camera trajectory CSV with fields frame,time_sec,x,y,z,"
            "roll_x_deg,pitch_y_deg,yaw_z_deg. Each row is rendered as one "
            "frame using its full camera pose."
        ),
    )
    parser.add_argument(
        "--camera_z",
        type=float,
        default=1.36,
        help="Camera height (m) for trajectory mode.",
    )
    parser.add_argument(
        "--camera_pitch",
        type=float,
        default=80.0,
        help="Camera pitch deg (rotation_deg[0]) for trajectory mode.",
    )
    parser.add_argument(
        "--camera_roll",
        type=float,
        default=0.0,
        help="Camera roll deg (rotation_deg[1]) for trajectory mode.",
    )
    parser.add_argument(
        "--glb_offset_xyz",
        type=float,
        nargs=3,
        default=(0.0012, 0.345, -1.36),
        metavar=("DX", "DY", "DZ"),
        help=(
            "GLB position minus camera position at yaw=0 (world frame). The "
            "horizontal part rotates with the camera yaw (rot); DZ is fixed."
        ),
    )
    parser.add_argument(
        "--glb_base_rotation_deg",
        type=float,
        nargs=3,
        default=(-90.0, 0.0, 90.0),
        metavar=("RX", "RY", "RZ"),
        help=(
            "GLB rotation deg at yaw=0; the Z component gets + rot per frame."
        ),
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Render every Nth waypoint/CSV row in trajectory mode.",
    )
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Render only these trajectory frame/waypoint indices. "
            "When set, --frame_stride is ignored."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-render trajectory frames even if they already exist "
            "(default: skip existing frames for per-frame resume)."
        ),
    )
    parser.add_argument(
        "--make_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stitch trajectory RGB frames into an mp4 (default: on).",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=0.0,
        help=(
            "Video frame rate. 0 = auto from JSON time_step or CSV time_sec "
            "and --frame_stride for real-time playback."
        ),
    )
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


class RenderUsd:
    """USD renderer for RGB, depth, normal, mesh, segmentation, and flow."""

    def __init__(
        self,
        *,
        usd_path: Path,
        glb_path: Path | str | None,
        glb_xyz: tuple[float, float, float] | list[float] | None,
        glb_rotation_deg: tuple[float, float, float] | list[float] | None,
        output_dir: Path,
        render_passes: tuple[str, ...] | list[str],
        depth_mode: str,
        depth_max: float,
        resolution: tuple[int, int] | list[int],
        samples: int,
        camera_xyz: tuple[float, float, float] | list[float],
        camera_rotation_deg: tuple[float, float, float] | list[float],
        flow_camera_xyz: tuple[float, float, float] | list[float] | None,
        flow_camera_rotation_deg: (
            tuple[float, float, float] | list[float] | None
        ),
        focal_length_mm: float,
        exposure: float,
        world_strength: float,
        fill_light_energy: float,
        remove_doors: bool = False,
    ) -> None:
        """Initialize renderer configuration independent of CLI parsing."""
        self.usd_path = usd_path
        self.glb_path = self.normalize_optional_path(glb_path)
        self.glb_xyz = tuple(glb_xyz) if glb_xyz is not None else None
        self.glb_rotation_deg = (
            list(glb_rotation_deg) if glb_rotation_deg is not None else None
        )
        if self.glb_rotation_deg is not None:
            self.glb_rotation_deg[0] += 90
        self.output_dir = output_dir
        self.render_passes = tuple(render_passes)
        self.depth_mode = depth_mode
        self.depth_max = depth_max
        self.resolution = tuple(resolution)
        self.samples = samples
        self.camera_xyz = tuple(camera_xyz)
        self.camera_rotation_deg = tuple(camera_rotation_deg)
        self.flow_camera_xyz = (
            tuple(flow_camera_xyz) if flow_camera_xyz is not None else None
        )
        self.flow_camera_rotation_deg = (
            tuple(flow_camera_rotation_deg)
            if flow_camera_rotation_deg is not None
            else None
        )
        self.focal_length_mm = focal_length_mm
        self.exposure = exposure
        self.world_strength = world_strength
        self.fill_light_energy = fill_light_energy
        self.remove_doors = remove_doors
        # Trajectory frame layout: route each pass to frames_<pass>/
        # frame_<idx>.<ext> under frame_base_dir instead of output_dir.
        self.frame_layout = False
        self.frame_base_dir: Path | None = None
        self.frame_index = 0
        self.temp_dir: Path | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RenderUsd:
        """Build a renderer from parsed CLI arguments."""
        return cls(
            usd_path=args.usd_path,
            glb_path=args.glb_path,
            glb_xyz=args.glb_xyz,
            glb_rotation_deg=args.glb_rotation_deg,
            output_dir=args.output_dir,
            render_passes=args.render_passes,
            depth_mode=args.depth_mode,
            depth_max=args.depth_max,
            resolution=args.resolution,
            samples=args.samples,
            camera_xyz=args.camera_xyz,
            camera_rotation_deg=args.camera_rotation_deg,
            flow_camera_xyz=args.flow_camera_xyz,
            flow_camera_rotation_deg=args.flow_camera_rotation_deg,
            focal_length_mm=args.focal_length_mm,
            exposure=args.exposure,
            world_strength=args.world_strength,
            fill_light_energy=args.fill_light_energy,
            remove_doors=args.remove_doors,
        )

    @property
    def scene(self) -> bpy.types.Scene:
        return bpy.context.scene

    def normalize_optional_path(
        self, path_value: Path | str | None
    ) -> Path | None:
        """Normalize an optional CLI path, treating empty strings as missing."""
        if path_value is None:
            return None
        if isinstance(path_value, Path):
            return path_value

        normalized = path_value.strip()
        if not normalized:
            return None
        return Path(normalized)

    def build_output_path(self, filename: str) -> Path:
        """Build a normalized output path under the render directory.

        In trajectory frame layout, each pass is routed to its own
        ``frames_<pass>/frame_<idx>.<ext>`` directory under
        ``frame_base_dir``; otherwise the file lives directly in
        ``output_dir``.
        """
        if self.frame_layout and self.frame_base_dir is not None:
            name = Path(filename)
            subdir = self.frame_base_dir / f"frames_{name.stem}"
            return subdir / f"frame_{self.frame_index:04d}{name.suffix}"
        return self.output_dir / filename

    def build_temp_path(self, filename: str) -> Path:
        """Build a temporary path outside the final output directory."""
        if self.temp_dir is None:
            raise RuntimeError(
                "Temporary render directory is not initialized."
            )
        return self.temp_dir / filename

    def get_rgb_output_path(self) -> Path:
        return self.build_output_path("render_rgb.png")

    def get_depth_vis_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_depth.png")

    def get_normal_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_normal.png")

    def get_mesh_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_mesh.png")

    def get_instance_seg_vis_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_instance_seg_vis.png")

    def get_instance_seg_temp_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_temp_path("render_instance_seg_raw_0001.exr")

    def get_flow_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_flow.npy")

    def get_flow_valid_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_flow_valid.npy")

    def get_flow_vis_output_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_output_path("render_flow_vis.png")

    def get_flow_depth_temp_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_temp_path("render_flow_depth_raw_0001.exr")

    def get_depth_gray_temp_path(self, output_path: Path) -> Path:
        del output_path
        return self.build_temp_path("render_depth_gray_0001.png")

    def get_composite_output_path(
        self, render_passes: list[str] | tuple[str, ...]
    ) -> Path:
        pass_names = "_".join(render_passes)
        return self.build_output_path(f"render_composite_{pass_names}.png")

    def build_occurrence_output_path(
        self, output_path: Path, occurrence_index: int
    ) -> Path:
        """Build an occurrence-specific path for repeated preview outputs."""
        if occurrence_index < 1:
            raise ValueError("occurrence_index must be greater than 0.")
        if occurrence_index == 1:
            return output_path

        return output_path.with_name(
            f"{output_path.stem}_{occurrence_index}{output_path.suffix}"
        )

    def iter_render_pass_occurrences(self) -> list[tuple[str, int]]:
        """Return requested render passes with 1-based occurrence indices."""
        occurrence_counts: dict[str, int] = {}
        render_pass_occurrences: list[tuple[str, int]] = []
        for render_pass_name in self.render_passes:
            occurrence_index = occurrence_counts.get(render_pass_name, 0) + 1
            occurrence_counts[render_pass_name] = occurrence_index
            render_pass_occurrences.append(
                (render_pass_name, occurrence_index)
            )
        return render_pass_occurrences

    def get_temp_output_slot_prefix(self, temp_output_path: Path) -> str:
        """Return the compositor slot prefix without the frame suffix."""
        stem_parts = temp_output_path.stem.rsplit("_", maxsplit=1)
        if len(stem_parts) != 2 or not stem_parts[1].isdigit():
            raise ValueError(
                f"Unexpected temporary output filename: {temp_output_path.name}"
            )
        return f"{stem_parts[0]}_"

    def get_mesh_objects(self) -> list[bpy.types.Object]:
        return [obj for obj in self.scene.objects if obj.type == "MESH"]

    def clear_scene(self) -> None:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def import_usd(self) -> None:
        if not self.usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {self.usd_path}")
        bpy.ops.wm.usd_import(filepath=str(self.usd_path))

    def remove_door_objects(self) -> int:
        """Delete any object whose name contains a door keyword."""
        removed = 0
        for obj in list(self.scene.objects):
            if any(kw in obj.name.lower() for kw in DOOR_KEYWORDS):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
        logger.info("Removed %d door objects.", removed)
        return removed

    def validate_glb_args(self, *, allow_path_only: bool = False) -> None:
        """Normalize optional GLB arguments and ensure all-or-none usage."""
        has_glb_path = self.glb_path is not None
        has_glb_xyz = self.glb_xyz is not None
        has_glb_rotation = self.glb_rotation_deg is not None
        if not has_glb_path:
            if has_glb_xyz or has_glb_rotation:
                raise ValueError(
                    "--glb_xyz and --glb_rotation_deg require --glb_path."
                )
            return
        if allow_path_only and not has_glb_xyz and not has_glb_rotation:
            self.validate_glb_path()
            return
        if not has_glb_xyz or not has_glb_rotation:
            raise ValueError(
                "--glb_path, --glb_xyz, and --glb_rotation_deg must be "
                "provided together, unless trajectory rendering follows the "
                "camera with --glb_offset_xyz/--glb_base_rotation_deg."
            )

        self.validate_glb_path()

    def validate_glb_path(self) -> None:
        """Ensure the optional GLB path points to an existing .glb file."""
        if self.glb_path is None:
            return
        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {self.glb_path}")
        if self.glb_path.suffix.lower() != ".glb":
            raise ValueError(
                f"Expected a .glb asset, but got: {self.glb_path}"
            )

    def enable_gltf_importer(self) -> None:
        """Ensure Blender's glTF importer add-on is available."""
        addon_name = "io_scene_gltf2"
        if addon_name in bpy.context.preferences.addons:
            return

        try:
            bpy.ops.preferences.addon_enable(module=addon_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to enable Blender glTF importer add-on."
            ) from exc

    def import_glb_asset(self) -> list[bpy.types.Object]:
        """Import the optional GLB asset and return created objects."""
        if self.glb_path is None:
            return []

        self.enable_gltf_importer()
        existing_object_ids = {obj.as_pointer() for obj in bpy.data.objects}
        result = bpy.ops.import_scene.gltf(filepath=str(self.glb_path))
        if "FINISHED" not in result:
            raise RuntimeError(f"Failed to import GLB asset: {self.glb_path}")

        imported_objects = [
            obj
            for obj in bpy.data.objects
            if obj.as_pointer() not in existing_object_ids
        ]
        if not imported_objects:
            raise ValueError(
                f"No objects were imported from GLB asset: {self.glb_path}"
            )
        return imported_objects

    def get_imported_root_objects(
        self, imported_objects: list[bpy.types.Object]
    ) -> list[bpy.types.Object]:
        """Return top-level imported objects so transforms apply as one asset."""
        imported_ids = {obj.as_pointer() for obj in imported_objects}
        root_objects = [
            obj
            for obj in imported_objects
            if obj.parent is None
            or obj.parent.as_pointer() not in imported_ids
        ]
        return root_objects or imported_objects

    def place_glb_asset(
        self, imported_objects: list[bpy.types.Object]
    ) -> None:
        """Place the imported GLB asset using the requested world transform."""
        if not imported_objects:
            return
        if self.glb_xyz is None or self.glb_rotation_deg is None:
            raise ValueError("GLB transform arguments are not initialized.")

        asset_transform = self.build_camera_matrix_world(
            self.glb_xyz,
            self.glb_rotation_deg,
        )
        for obj in self.get_imported_root_objects(imported_objects):
            obj.matrix_world = asset_transform @ obj.matrix_world.copy()
        bpy.context.view_layer.update()

    def get_scene_bbox(self) -> tuple[Vector, Vector]:
        """Compute the world-space bounding box across all mesh objects."""
        mesh_objects = self.get_mesh_objects()
        if not mesh_objects:
            raise ValueError("No mesh objects found after USD import.")

        points: list[Vector] = []
        for obj in mesh_objects:
            points.extend(
                obj.matrix_world @ Vector(corner) for corner in obj.bound_box
            )

        min_corner = Vector(
            (
                min(p.x for p in points),
                min(p.y for p in points),
                min(p.z for p in points),
            )
        )
        max_corner = Vector(
            (
                max(p.x for p in points),
                max(p.y for p in points),
                max(p.z for p in points),
            )
        )
        return min_corner, max_corner

    def create_camera(self) -> bpy.types.Object:
        """Create and configure the primary render camera."""
        if self.camera_xyz is None:
            raise ValueError("--camera_xyz is required.")

        location = Vector(tuple(self.camera_xyz))
        rotation_rad = self.get_rotation_radians(self.camera_rotation_deg)
        bpy.ops.object.camera_add(location=location, rotation=rotation_rad)
        camera = bpy.context.object
        camera.rotation_mode = "XYZ"
        camera.data.lens = self.focal_length_mm
        camera.data.clip_start = 0.01
        camera.data.clip_end = 1000.0
        self.scene.camera = camera
        return camera

    def add_light_rig(
        self,
        diagonal: float,
        center: Vector,
        top_z: float,
        *,
        area_energy: float,
        sun_energy: float,
        prefix: str,
    ) -> None:
        bpy.ops.object.light_add(
            type="AREA",
            location=(center.x, center.y, top_z + 0.5 * diagonal),
        )
        area = bpy.context.object
        area.name = f"{prefix}Area"
        area.data.energy = area_energy
        area.data.shape = "DISK"
        area.data.size = max(diagonal, 2.0)

        bpy.ops.object.light_add(
            type="SUN",
            location=(
                center.x + diagonal,
                center.y - diagonal,
                top_z + diagonal,
            ),
        )
        sun = bpy.context.object
        sun.name = f"{prefix}Sun"
        sun.data.energy = sun_energy

    def add_fill_light(
        self,
        diagonal: float,
        center: Vector,
        top_z: float,
        energy: float,
    ) -> None:
        if energy <= 0.0:
            return

        bpy.ops.object.light_add(
            type="AREA",
            location=(center.x, center.y, top_z + 0.35 * diagonal),
            rotation=(0.0, 0.0, 0.0),
        )
        area = bpy.context.object
        area.name = "GlobalFillArea"
        area.data.energy = energy
        area.data.shape = "DISK"
        area.data.size = max(diagonal * 0.9, 3.0)

    def ensure_lighting(
        self, diagonal: float, center: Vector, top_z: float
    ) -> None:
        if any(obj.type == "LIGHT" for obj in self.scene.objects):
            return

        self.add_light_rig(
            diagonal,
            center,
            top_z,
            area_energy=5000.0,
            sun_energy=1.5,
            prefix="Fallback",
        )

    def set_world_strength(self, strength: float) -> None:
        world = self.scene.world
        if world is None:
            return

        if not world.use_nodes:
            world.use_nodes = True

        tree = world.node_tree
        background_nodes = [
            node for node in tree.nodes if node.type == "BACKGROUND"
        ]
        if not background_nodes:
            background = tree.nodes.new(type="ShaderNodeBackground")
            output = next(
                (node for node in tree.nodes if node.type == "OUTPUT_WORLD"),
                None,
            )
            if output is None:
                output = tree.nodes.new(type="ShaderNodeOutputWorld")
            tree.links.new(
                background.outputs["Background"], output.inputs["Surface"]
            )
            background_nodes = [background]

        for background in background_nodes:
            background.inputs["Strength"].default_value = strength

    def ensure_world(self) -> bool:
        """Ensure the scene has a world shader and return whether it was created."""
        if self.scene.world is not None:
            self.set_world_strength(self.world_strength)
            return False

        world = bpy.data.worlds.new(name="RenderWorld")
        world.use_nodes = True
        tree = world.node_tree
        tree.nodes.clear()

        output = tree.nodes.new(type="ShaderNodeOutputWorld")
        background = tree.nodes.new(type="ShaderNodeBackground")
        sky = tree.nodes.new(type="ShaderNodeTexSky")

        background.inputs["Strength"].default_value = self.world_strength

        tree.links.new(sky.outputs["Color"], background.inputs["Color"])
        tree.links.new(
            background.outputs["Background"], output.inputs["Surface"]
        )

        self.scene.world = world
        return True

    def configure_cycles(self) -> None:
        self.scene.render.engine = "CYCLES"
        self.scene.cycles.device = "GPU"
        self.scene.cycles.samples = self.samples
        self.scene.render.resolution_x = self.resolution[0]
        self.scene.render.resolution_y = self.resolution[1]
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.film_transparent = False
        self.scene.render.use_persistent_data = False

        # Adaptive sampling + denoising keep quality high at low sample counts.
        self.scene.cycles.use_adaptive_sampling = True
        self.scene.cycles.use_denoising = True

        self.enable_gpu_devices()

    def enable_gpu_devices(self) -> None:
        """Enable Cycles GPU devices, preferring OptiX then CUDA."""
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            prefs.get_devices()
            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if not gpu_devices:
                continue

            for device in prefs.devices:
                device.use = device.type == backend
            try:
                self.scene.cycles.denoiser = (
                    "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
                )
            except (TypeError, AttributeError):
                pass
            logger.info(
                "Cycles GPU backend: %s (%d device(s)).",
                backend,
                len(gpu_devices),
            )
            return

        raise RuntimeError("No OptiX/CUDA GPU device found in Blender Cycles.")

    def configure_color_management(self) -> None:
        self.scene.view_settings.exposure = self.exposure

    def snapshot_render_state(
        self,
        view_layer: bpy.types.ViewLayer,
        *,
        include_filepath: bool = False,
        include_material_override: bool = False,
        include_use_pass_z: bool = False,
        include_use_pass_object_index: bool = False,
    ) -> dict[str, object]:
        """Capture the render state that temporary passes need to restore."""
        state: dict[str, object] = {
            "film_transparent": self.scene.render.film_transparent,
            "view_transform": self.scene.view_settings.view_transform,
            "look": self.scene.view_settings.look,
            "exposure": self.scene.view_settings.exposure,
            "gamma": self.scene.view_settings.gamma,
            "file_format": self.scene.render.image_settings.file_format,
            "color_mode": self.scene.render.image_settings.color_mode,
            "color_depth": self.scene.render.image_settings.color_depth,
            "use_nodes": self.scene.use_nodes,
            "samples": self.scene.cycles.samples,
            "use_denoising": self.scene.cycles.use_denoising,
        }
        if include_filepath:
            state["filepath"] = self.scene.render.filepath
        if include_material_override:
            state["material_override"] = view_layer.material_override
        if include_use_pass_z:
            state["use_pass_z"] = view_layer.use_pass_z
        if include_use_pass_object_index:
            state["use_pass_object_index"] = view_layer.use_pass_object_index
        return state

    def restore_render_state(
        self, state: dict[str, object], view_layer: bpy.types.ViewLayer
    ) -> None:
        """Restore a render state captured by ``snapshot_render_state``."""
        self.scene.render.film_transparent = state["film_transparent"]
        self.scene.view_settings.view_transform = state["view_transform"]
        self.scene.view_settings.look = state["look"]
        self.scene.view_settings.exposure = state["exposure"]
        self.scene.view_settings.gamma = state["gamma"]
        self.scene.render.image_settings.file_format = state["file_format"]
        self.scene.render.image_settings.color_mode = state["color_mode"]
        self.scene.render.image_settings.color_depth = state["color_depth"]
        self.scene.use_nodes = state["use_nodes"]
        self.scene.cycles.samples = state["samples"]
        self.scene.cycles.use_denoising = state["use_denoising"]
        if "filepath" in state:
            self.scene.render.filepath = state["filepath"]
        if "material_override" in state:
            view_layer.material_override = state["material_override"]
        if "use_pass_z" in state:
            view_layer.use_pass_z = state["use_pass_z"]
        if "use_pass_object_index" in state:
            view_layer.use_pass_object_index = state["use_pass_object_index"]

    def apply_raw_preview_settings(
        self,
        *,
        use_nodes: bool,
        samples: int,
        color_mode: str,
        color_depth: str,
    ) -> None:
        """Apply the shared render settings for auxiliary preview passes."""
        self.scene.render.film_transparent = True
        self.scene.view_settings.view_transform = "Raw"
        self.scene.view_settings.look = "None"
        self.scene.view_settings.exposure = 0.0
        self.scene.view_settings.gamma = 1.0
        self.scene.use_nodes = use_nodes
        self.scene.cycles.samples = samples
        # Auxiliary passes (normal/mesh/seg/flow depth) must not be denoised:
        # denoising would blur normals and corrupt integer index passes.
        self.scene.cycles.use_denoising = False
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.image_settings.color_mode = color_mode
        self.scene.render.image_settings.color_depth = color_depth

    def clear_compositor_tree(self) -> bpy.types.NodeTree:
        """Reset the compositor tree so each pass starts from a clean slate."""
        self.scene.use_nodes = True
        tree = self.scene.node_tree
        tree.nodes.clear()
        return tree

    def remove_render_nodes(self, created_nodes: list[bpy.types.Node]) -> None:
        """Remove compositor nodes created for a temporary render pass."""
        if not created_nodes:
            return

        node_tree = self.scene.node_tree
        if node_tree is None:
            return

        for node in created_nodes:
            if node.name in node_tree.nodes:
                node_tree.nodes.remove(node)

    def render_material_override_pass(
        self,
        preview_output_path: Path,
        material_factory: Callable[[], bpy.types.Material],
        *,
        color_mode: str,
    ) -> None:
        """Render a pass with a temporary material override."""
        preview_output_path.parent.mkdir(parents=True, exist_ok=True)
        view_layer = self.scene.view_layers["ViewLayer"]
        state = self.snapshot_render_state(
            view_layer,
            include_filepath=True,
            include_material_override=True,
        )

        material = material_factory()
        try:
            self.apply_raw_preview_settings(
                use_nodes=False,
                samples=min(int(state["samples"]), 64),
                color_mode=color_mode,
                color_depth="8",
            )
            self.scene.render.filepath = str(preview_output_path)
            view_layer.material_override = material
            bpy.ops.render.render(write_still=True)
        finally:
            self.restore_render_state(state, view_layer)
            bpy.data.materials.remove(material, do_unlink=True)

    def render_temp_output_pass(
        self,
        output_path: Path,
        temp_output_path: Path,
        *,
        add_output_node: Callable[
            [Path], tuple[bpy.types.NodeTree, list[bpy.types.Node]]
        ],
        load_temp_output: Callable[[Path], np.ndarray],
        finalize_output: Callable[[np.ndarray], None],
        color_mode: str,
        color_depth: str,
        enable_depth_pass: bool = False,
        enable_object_index_pass: bool = False,
    ) -> None:
        """Render a temporary compositor output and finalize it."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        view_layer = self.scene.view_layers["ViewLayer"]
        state = self.snapshot_render_state(
            view_layer,
            include_use_pass_z=enable_depth_pass,
            include_use_pass_object_index=enable_object_index_pass,
        )
        created_nodes: list[bpy.types.Node] = []

        try:
            if temp_output_path.exists():
                temp_output_path.unlink()

            self.apply_raw_preview_settings(
                use_nodes=True,
                samples=1,
                color_mode=color_mode,
                color_depth=color_depth,
            )
            if enable_depth_pass:
                view_layer.use_pass_z = True
            if enable_object_index_pass:
                view_layer.use_pass_object_index = True

            self.clear_compositor_tree()
            _, created_nodes = add_output_node(output_path)
            bpy.ops.render.render(write_still=False)
            finalize_output(load_temp_output(temp_output_path))
        finally:
            self.remove_render_nodes(created_nodes)
            if temp_output_path.exists():
                temp_output_path.unlink()
            self.restore_render_state(state, view_layer)

    def get_rotation_radians(
        self, rotation_deg: tuple[float, float, float] | list[float]
    ) -> tuple[float, float, float]:
        return tuple(math.radians(angle_deg) for angle_deg in rotation_deg)

    def validate_flow_args(self) -> None:
        """Normalize optional flow-camera arguments and fill defaults."""
        has_flow_xyz = self.flow_camera_xyz is not None
        has_flow_rotation = self.flow_camera_rotation_deg is not None
        if has_flow_xyz != has_flow_rotation:
            raise ValueError(
                "--flow_camera_xyz and --flow_camera_rotation_deg must be "
                "provided together."
            )
        if not has_flow_xyz:
            xyz = list(self.camera_xyz)
            xyz[0] += 0.5
            self.flow_camera_xyz = tuple(xyz)
            self.flow_camera_rotation_deg = tuple(self.camera_rotation_deg)

    def build_depth_preview_node(
        self,
        tree: bpy.types.NodeTree,
        render_layers: bpy.types.CompositorNodeRLayers,
        camera: bpy.types.Camera,
        depth_mode: str,
    ) -> bpy.types.Node:
        """Build the compositor node that converts raw depth to a previewable map."""
        del camera  # Mapping uses self.depth_max, not the camera clip range.
        if depth_mode == "normalized":
            normalize = tree.nodes.new(type="CompositorNodeNormalize")
            tree.links.new(render_layers.outputs["Depth"], normalize.inputs[0])
            return normalize

        if depth_mode != "metric":
            raise ValueError(f"Unsupported depth mode: {depth_mode}")

        depth_map = tree.nodes.new(type="CompositorNodeMapRange")
        # Fixed 0..depth_max mapping keeps depth consistent across frames.
        depth_map.inputs["From Min"].default_value = 0.0
        depth_map.inputs["From Max"].default_value = self.depth_max
        depth_map.inputs["To Min"].default_value = 0.0
        depth_map.inputs["To Max"].default_value = 1.0
        depth_map.use_clamp = True
        tree.links.new(render_layers.outputs["Depth"], depth_map.inputs[0])
        return depth_map

    def build_depth_vis_output(
        self,
        tree: bpy.types.NodeTree,
        depth_preview_node: bpy.types.Node,
        output_path: Path,
    ) -> Path:
        temp_output_path = self.get_depth_gray_temp_path(output_path)
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = str(temp_output_path.parent)
        output_node.file_slots[0].path = self.get_temp_output_slot_prefix(
            temp_output_path
        )
        output_node.format.file_format = "PNG"
        output_node.format.color_mode = "BW"
        output_node.format.color_depth = "8"

        tree.links.new(depth_preview_node.outputs[0], output_node.inputs[0])
        return temp_output_path

    def configure_auxiliary_outputs(
        self,
        output_path: Path,
        render_passes: tuple[str, ...] | list[str],
        depth_mode: str,
    ) -> list[tuple[Path, Path]]:
        """Configure compositor outputs needed during the base render."""
        view_layer = self.scene.view_layers["ViewLayer"]
        if "depth" in render_passes:
            view_layer.use_pass_z = True

        if "depth" not in render_passes:
            return []

        tree = self.clear_compositor_tree()

        render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        temp_outputs: list[tuple[Path, Path]] = []

        depth_preview_node = self.build_depth_preview_node(
            tree,
            render_layers,
            self.scene.camera.data,
            depth_mode,
        )
        temp_path = self.build_depth_vis_output(
            tree=tree,
            depth_preview_node=depth_preview_node,
            output_path=output_path,
        )
        temp_outputs.append(
            (temp_path, self.get_depth_vis_output_path(output_path))
        )

        return temp_outputs

    def finalize_depth_output(
        self, temp_path: Path, output_path: Path
    ) -> None:
        """Convert the grayscale depth temp image into the final colored preview."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        if not temp_path.exists():
            raise FileNotFoundError(f"Depth file not generated: {temp_path}")
        try:
            depth = cv2.imread(str(temp_path), cv2.IMREAD_GRAYSCALE)
            if depth is None:
                raise FileNotFoundError(
                    f"Failed to read depth image: {temp_path}"
                )

            depth_uint8 = np.ascontiguousarray(depth)
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

            if not cv2.imwrite(str(output_path), depth_colormap):
                raise RuntimeError(
                    f"Failed to write depth visualization: {output_path}"
                )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def create_clean_material(self, material_name: str) -> bpy.types.Material:
        """Create a material with a cleared node tree."""
        existing = bpy.data.materials.get(material_name)
        if existing is not None:
            bpy.data.materials.remove(existing, do_unlink=True)

        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        material.shadow_method = "NONE"
        tree = material.node_tree
        tree.nodes.clear()
        return material

    def create_view_normal_material(self) -> bpy.types.Material:
        material = self.create_clean_material("EmbodiedGenViewNormal")
        tree = material.node_tree

        geometry = tree.nodes.new(type="ShaderNodeNewGeometry")
        invert = tree.nodes.new(type="ShaderNodeVectorMath")
        invert.operation = "MULTIPLY"
        invert.inputs[1].default_value = (-1.0, -1.0, -1.0)

        face_mix = tree.nodes.new(type="ShaderNodeMix")
        face_mix.data_type = "VECTOR"
        face_mix.clamp_factor = True
        face_mix.factor_mode = "UNIFORM"

        view_transform = tree.nodes.new(type="ShaderNodeVectorTransform")
        view_transform.vector_type = "NORMAL"
        view_transform.convert_from = "WORLD"
        view_transform.convert_to = "CAMERA"

        flip_x = tree.nodes.new(type="ShaderNodeVectorMath")
        flip_x.operation = "MULTIPLY"
        flip_x.inputs[1].default_value = (-1.0, 1.0, -1.0)

        scale_bias = tree.nodes.new(type="ShaderNodeVectorMath")
        scale_bias.operation = "MULTIPLY_ADD"
        scale_bias.inputs[1].default_value = (0.5, 0.5, 0.5)
        scale_bias.inputs[2].default_value = (0.5, 0.5, 0.5)

        emission = tree.nodes.new(type="ShaderNodeEmission")
        output = tree.nodes.new(type="ShaderNodeOutputMaterial")

        tree.links.new(geometry.outputs["True Normal"], invert.inputs[0])
        tree.links.new(
            geometry.outputs["Backfacing"], face_mix.inputs["Factor"]
        )
        tree.links.new(geometry.outputs["True Normal"], face_mix.inputs["A"])
        tree.links.new(invert.outputs["Vector"], face_mix.inputs["B"])
        tree.links.new(
            face_mix.outputs["Result"], view_transform.inputs["Vector"]
        )
        tree.links.new(view_transform.outputs["Vector"], flip_x.inputs[0])
        tree.links.new(flip_x.outputs["Vector"], scale_bias.inputs[0])
        tree.links.new(scale_bias.outputs["Vector"], emission.inputs["Color"])
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

        return material

    def create_mesh_preview_material(self) -> bpy.types.Material:
        material = self.create_clean_material("EmbodiedGenMeshPreview")
        tree = material.node_tree

        layer_weight = tree.nodes.new(type="ShaderNodeLayerWeight")
        layer_weight.inputs["Blend"].default_value = 0.35

        base_ramp = tree.nodes.new(type="ShaderNodeValToRGB")
        base_ramp.color_ramp.elements[0].position = 0.1
        base_ramp.color_ramp.elements[0].color = (0.78, 0.81, 0.87, 1.0)
        base_ramp.color_ramp.elements[1].position = 0.9
        base_ramp.color_ramp.elements[1].color = (0.42, 0.48, 0.58, 1.0)

        emission = tree.nodes.new(type="ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 0.82
        output = tree.nodes.new(type="ShaderNodeOutputMaterial")

        tree.links.new(layer_weight.outputs["Facing"], base_ramp.inputs["Fac"])
        tree.links.new(base_ramp.outputs["Color"], emission.inputs["Color"])
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

        return material

    def assign_instance_ids(self) -> dict[str, int]:
        """Assign stable per-object pass indices for instance segmentation."""
        mesh_objects = sorted(
            self.get_mesh_objects(), key=lambda obj: obj.name
        )
        if not mesh_objects:
            raise ValueError(
                "No mesh objects found for instance segmentation."
            )

        instance_id_map: dict[str, int] = {}
        for instance_id, obj in enumerate(mesh_objects, start=1):
            obj.pass_index = instance_id
            instance_id_map[obj.name] = instance_id
        return instance_id_map

    def snapshot_object_pass_indices(
        self,
    ) -> list[tuple[bpy.types.Object, int]]:
        """Capture original object pass indices before a temporary override."""
        return [(obj, obj.pass_index) for obj in self.get_mesh_objects()]

    def restore_object_pass_indices(
        self, original_pass_indices: list[tuple[bpy.types.Object, int]]
    ) -> None:
        """Restore object pass indices captured earlier."""
        for obj, pass_index in original_pass_indices:
            obj.pass_index = pass_index

    def add_instance_seg_output_node(
        self,
        output_path: Path,
    ) -> tuple[bpy.types.NodeTree, list[bpy.types.Node]]:
        return self.add_exr_output_node(
            output_path=output_path,
            temp_output_path=self.get_instance_seg_temp_path(output_path),
            render_output_name="IndexOB",
        )

    def add_flow_depth_output_node(
        self,
        output_path: Path,
    ) -> tuple[bpy.types.NodeTree, list[bpy.types.Node]]:
        return self.add_exr_output_node(
            output_path=output_path,
            temp_output_path=self.get_flow_depth_temp_path(output_path),
            render_output_name="Depth",
        )

    def add_exr_output_node(
        self,
        output_path: Path,
        temp_output_path: Path,
        render_output_name: str,
    ) -> tuple[bpy.types.NodeTree, list[bpy.types.Node]]:
        """Attach a file-output EXR node for a specific render-layer socket."""
        tree = self.scene.node_tree
        render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = str(temp_output_path.parent)
        output_node.file_slots[0].path = self.get_temp_output_slot_prefix(
            temp_output_path
        )
        output_node.format.file_format = "OPEN_EXR"
        output_node.format.color_mode = "RGB"
        output_node.format.color_depth = "32"
        output_node.format.exr_codec = "NONE"

        tree.links.new(
            render_layers.outputs[render_output_name], output_node.inputs[0]
        )
        return tree, [render_layers, output_node]

    def load_temp_exr_first_channel(
        self,
        temp_path: Path,
        error_message: str,
    ) -> np.ndarray:
        """Load the first channel from a temporary EXR and flip to image space."""
        if not temp_path.exists():
            raise FileNotFoundError(error_message.format(path=temp_path))

        temp_image = bpy.data.images.load(str(temp_path), check_existing=False)
        try:
            width, height = temp_image.size
            channels = temp_image.channels
            pixels = np.array(temp_image.pixels[:], dtype=np.float32)
            if pixels.size != width * height * channels:
                raise RuntimeError(
                    f"Unexpected EXR image layout for {temp_path}."
                )

            image = pixels.reshape(height, width, channels)[..., 0]
            return np.flipud(image)
        finally:
            bpy.data.images.remove(temp_image)

    def load_instance_seg_temp_output(self, temp_path: Path) -> np.ndarray:
        instance_seg = self.load_temp_exr_first_channel(
            temp_path,
            "Instance segmentation file not generated: {path}",
        )
        return np.ascontiguousarray(np.rint(instance_seg).astype(np.uint16))

    def load_flow_depth_temp_output(self, temp_path: Path) -> np.ndarray:
        depth = self.load_temp_exr_first_channel(
            temp_path,
            "Flow depth file not generated: {path}",
        )
        depth = np.ascontiguousarray(depth.astype(np.float32))
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def build_instance_seg_visualization(
        self, instance_seg: np.ndarray, max_instance_id: int
    ) -> np.ndarray:
        """Map instance ids to deterministic RGB colors for visualization."""
        color_lut = np.zeros((max_instance_id + 1, 3), dtype=np.uint8)
        for instance_id in range(1, max_instance_id + 1):
            color_lut[instance_id] = (
                (instance_id * 37) % 256,
                (instance_id * 67) % 256,
                (instance_id * 97) % 256,
            )
        return color_lut[instance_seg]

    def save_instance_seg_outputs(
        self,
        output_path: Path,
        instance_seg: np.ndarray,
    ) -> None:
        vis_output_path = self.get_instance_seg_vis_output_path(output_path)
        vis_output_path.parent.mkdir(parents=True, exist_ok=True)

        visualization = self.build_instance_seg_visualization(
            instance_seg=instance_seg,
            max_instance_id=int(instance_seg.max(initial=0)),
        )
        if not cv2.imwrite(str(vis_output_path), visualization):
            raise RuntimeError(
                f"Failed to write instance segmentation preview: "
                f"{vis_output_path}"
            )

    def build_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        flow_float = flow.astype(np.float32)
        magnitude, angle = cv2.cartToPolar(
            flow_float[..., 0],
            flow_float[..., 1],
            angleInDegrees=True,
        )
        max_magnitude = float(np.percentile(magnitude, 99.0))
        if max_magnitude <= 1e-6:
            max_magnitude = 1.0

        magnitude_norm = np.clip(magnitude / max_magnitude, 0.0, 1.0)
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
        hsv[..., 0] = np.mod(angle, 360.0)
        hsv[..., 1] = magnitude_norm
        hsv[..., 2] = 1.0
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return np.clip(bgr * 255.0, 0.0, 255.0).astype(np.uint8)

    def get_camera_intrinsics(
        self, camera: bpy.types.Object, width: int, height: int
    ) -> tuple[float, float, float, float]:
        camera_data = camera.data
        fx = width / (2.0 * math.tan(camera_data.angle_x * 0.5))
        fy = height / (2.0 * math.tan(camera_data.angle_y * 0.5))
        cx = (width - 1.0) * 0.5
        cy = (height - 1.0) * 0.5
        return fx, fy, cx, cy

    def build_camera_matrix_world(
        self,
        xyz: tuple[float, float, float] | list[float],
        rotation_deg: tuple[float, float, float] | list[float],
    ) -> Matrix:
        rotation = Euler(self.get_rotation_radians(rotation_deg), "XYZ")
        translation = Matrix.Translation(Vector(tuple(xyz)))
        return translation @ rotation.to_matrix().to_4x4()

    def compute_flow_from_depth(
        self,
        depth: np.ndarray,
        camera: bpy.types.Object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project depth into a target camera and derive dense 2D flow."""
        height, width = depth.shape
        fx, fy, cx, cy = self.get_camera_intrinsics(camera, width, height)
        valid = np.isfinite(depth) & (depth > 0.0)
        valid_mask = np.zeros((height, width), dtype=bool)
        if not np.any(valid):
            return np.zeros((height, width, 2), dtype=np.float32), valid_mask

        u_coords, v_coords = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )

        depth_valid = depth[valid]
        x_cam = ((u_coords[valid] - cx) / fx) * depth_valid
        y_cam = (-(v_coords[valid] - cy) / fy) * depth_valid
        z_cam = -depth_valid

        camera_points = np.stack(
            [x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1
        )

        source_matrix_world = np.array(camera.matrix_world, dtype=np.float64)
        target_matrix_world = np.array(
            self.build_camera_matrix_world(
                self.flow_camera_xyz,
                self.flow_camera_rotation_deg,
            ),
            dtype=np.float64,
        )
        target_world_to_camera = np.linalg.inv(target_matrix_world)

        world_points = camera_points @ source_matrix_world.T
        target_camera_points = world_points @ target_world_to_camera.T

        target_z = target_camera_points[:, 2]
        positive_depth = target_z < -1e-6
        flow = np.zeros((height, width, 2), dtype=np.float32)
        if not np.any(positive_depth):
            return flow, valid_mask

        projected_x = (
            fx
            * (
                target_camera_points[positive_depth, 0]
                / -target_z[positive_depth]
            )
            + cx
        )
        projected_y = (
            -fy
            * (
                target_camera_points[positive_depth, 1]
                / -target_z[positive_depth]
            )
            + cy
        )
        in_frame = (
            (projected_x >= 0.0)
            & (projected_x < width)
            & (projected_y >= 0.0)
            & (projected_y < height)
        )
        if not np.any(in_frame):
            return flow, valid_mask

        source_x = u_coords[valid][positive_depth]
        source_y = v_coords[valid][positive_depth]
        flow_valid = np.stack(
            [
                projected_x[in_frame] - source_x[in_frame],
                projected_y[in_frame] - source_y[in_frame],
            ],
            axis=1,
        ).astype(np.float32)

        flow_buffer = flow[valid]
        positive_depth_buffer = flow_buffer[positive_depth]
        positive_depth_buffer[in_frame] = flow_valid
        flow_buffer[positive_depth] = positive_depth_buffer
        flow[valid] = flow_buffer
        valid_mask_buffer = valid_mask[valid]
        positive_depth_mask = valid_mask_buffer[positive_depth]
        positive_depth_mask[in_frame] = True
        valid_mask_buffer[positive_depth] = positive_depth_mask
        valid_mask[valid] = valid_mask_buffer
        return flow, valid_mask

    def save_numpy_array(self, output_path: Path, array: np.ndarray) -> None:
        """Persist a NumPy array atomically to avoid partial writes."""
        temp_output_path = output_path.with_suffix(".tmp.npy")
        if temp_output_path.exists():
            temp_output_path.unlink()
        np.save(temp_output_path, array)
        temp_output_path.replace(output_path)

    def save_flow_outputs(
        self,
        output_path: Path,
        flow: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        flow_output_path = self.get_flow_output_path(output_path)
        flow_valid_output_path = self.get_flow_valid_output_path(output_path)
        flow_vis_output_path = self.get_flow_vis_output_path(output_path)
        for path in (
            flow_output_path,
            flow_valid_output_path,
            flow_vis_output_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        self.save_numpy_array(flow_output_path, flow)
        self.save_numpy_array(flow_valid_output_path, valid_mask)
        flow_vis = self.build_flow_visualization(flow)
        if not cv2.imwrite(str(flow_vis_output_path), flow_vis):
            raise RuntimeError(
                f"Failed to write flow preview: {flow_vis_output_path}"
            )

    def get_preview_output_path(
        self,
        output_path: Path,
        render_pass_name: str,
        occurrence_index: int = 1,
    ) -> Path | None:
        preview_output_paths = {
            "rgb": output_path,
            "depth": self.get_depth_vis_output_path(output_path),
            "normal": self.get_normal_output_path(output_path),
            "mesh": self.get_mesh_output_path(output_path),
            "instance_seg": self.get_instance_seg_vis_output_path(output_path),
            "flow": self.get_flow_vis_output_path(output_path),
        }
        preview_output_path = preview_output_paths.get(render_pass_name)
        if preview_output_path is None:
            return None
        return self.build_occurrence_output_path(
            preview_output_path, occurrence_index
        )

    def load_preview_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"Failed to read preview image: {image_path}"
            )
        return image

    def collect_composite_images(
        self, output_path: Path
    ) -> list[tuple[str, np.ndarray]]:
        composite_images: list[tuple[str, np.ndarray]] = []
        for (
            render_pass_name,
            occurrence_index,
        ) in self.iter_render_pass_occurrences():
            preview_output_path = self.get_preview_output_path(
                output_path,
                render_pass_name,
                occurrence_index,
            )
            if preview_output_path is None or not preview_output_path.exists():
                continue
            composite_images.append(
                (
                    render_pass_name,
                    self.load_preview_image(preview_output_path),
                )
            )
        return composite_images

    def replicate_duplicate_preview_outputs(self, output_path: Path) -> None:
        """Materialize repeated preview outputs without re-rendering."""
        for (
            render_pass_name,
            occurrence_index,
        ) in self.iter_render_pass_occurrences():
            if occurrence_index == 1:
                continue

            source_output_path = self.get_preview_output_path(
                output_path, render_pass_name
            )
            duplicate_output_path = self.get_preview_output_path(
                output_path,
                render_pass_name,
                occurrence_index,
            )
            if source_output_path is None or duplicate_output_path is None:
                continue
            if not source_output_path.exists():
                raise FileNotFoundError(
                    f"Preview output not generated for repeated pass "
                    f"{render_pass_name}: {source_output_path}"
                )
            if duplicate_output_path.exists():
                duplicate_output_path.unlink()
            shutil.copyfile(source_output_path, duplicate_output_path)

    def get_composite_separator_boundaries(
        self,
        render_pass_names: list[str] | tuple[str, ...],
        boundaries: np.ndarray,
    ) -> list[float]:
        """Return separator boundaries for adjacent passes that differ."""
        if len(boundaries) != len(render_pass_names) + 1:
            raise ValueError(
                "boundaries length must match the number of render passes + 1."
            )

        separator_boundaries: list[float] = []
        for index, boundary in enumerate(boundaries[1:-1], start=1):
            if render_pass_names[index - 1] == render_pass_names[index]:
                continue
            separator_boundaries.append(float(boundary))
        return separator_boundaries

    def build_composite_image(
        self,
        images: list[np.ndarray],
        render_pass_names: list[str] | tuple[str, ...],
        separator_width_px: int = 6,
    ) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required for composition.")
        if len(images) != len(render_pass_names):
            raise ValueError(
                "images and render_pass_names must have the same length."
            )

        base_height, base_width = images[0].shape[:2]
        resized_images = [
            (
                image
                if image.shape[:2] == (base_height, base_width)
                else cv2.resize(
                    image,
                    (base_width, base_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            for image in images
        ]

        x_coords = np.broadcast_to(
            np.arange(base_width, dtype=np.float32),
            (base_height, base_width),
        )
        y_coords = np.broadcast_to(
            np.arange(base_height, dtype=np.float32)[:, None],
            (base_height, base_width),
        )
        slash_slope = 0.28 * (base_width / base_height)
        diagonal_coord = x_coords + y_coords * slash_slope
        diagonal_min = float(diagonal_coord.min())
        diagonal_max = float(diagonal_coord.max())
        boundaries = np.linspace(
            diagonal_min, diagonal_max, len(resized_images) + 1
        )

        composite = np.zeros_like(resized_images[0])
        region_indices = np.digitize(
            diagonal_coord, boundaries[1:-1], right=False
        )
        for image_index, image in enumerate(resized_images):
            composite[region_indices == image_index] = image[
                region_indices == image_index
            ]

        slash_mask = np.zeros((base_height, base_width), dtype=bool)
        separator_boundaries = self.get_composite_separator_boundaries(
            render_pass_names, boundaries
        )
        for boundary in separator_boundaries:
            slash_mask |= (
                np.abs(diagonal_coord - boundary) <= separator_width_px
            )
        composite[slash_mask] = 255
        return composite

    def save_composite_preview(self, output_path: Path) -> None:
        composite_images = self.collect_composite_images(output_path)
        if len(composite_images) < 2:
            return

        composite_output_path = self.get_composite_output_path(
            tuple(render_pass_name for render_pass_name, _ in composite_images)
        )
        composite_output_path.parent.mkdir(parents=True, exist_ok=True)
        composite_image = self.build_composite_image(
            [image for _, image in composite_images],
            [render_pass_name for render_pass_name, _ in composite_images],
        )
        if not cv2.imwrite(str(composite_output_path), composite_image):
            raise RuntimeError(
                f"Failed to write composite preview: {composite_output_path}"
            )

    def render_flow_pass(self, output_path: Path) -> None:
        self.validate_flow_args()
        camera = self.scene.camera
        if camera is None:
            raise ValueError("Scene camera is required for flow rendering.")

        temp_output_path = self.get_flow_depth_temp_path(output_path)

        def finalize_flow_output(depth: np.ndarray) -> None:
            flow, valid_mask = self.compute_flow_from_depth(
                depth=depth, camera=camera
            )
            self.save_flow_outputs(
                output_path=output_path,
                flow=flow,
                valid_mask=valid_mask,
            )

        self.render_temp_output_pass(
            output_path=output_path,
            temp_output_path=temp_output_path,
            add_output_node=self.add_flow_depth_output_node,
            load_temp_output=self.load_flow_depth_temp_output,
            finalize_output=finalize_flow_output,
            color_mode="RGB",
            color_depth="8",
            enable_depth_pass=True,
        )

    def render_normal_pass(self, output_path: Path) -> None:
        normal_output_path = self.get_normal_output_path(output_path)
        self.render_material_override_pass(
            preview_output_path=normal_output_path,
            material_factory=self.create_view_normal_material,
            color_mode="RGB",
        )

    def render_mesh_pass(self, output_path: Path) -> None:
        mesh_output_path = self.get_mesh_output_path(output_path)
        self.render_material_override_pass(
            preview_output_path=mesh_output_path,
            material_factory=self.create_mesh_preview_material,
            color_mode="RGBA",
        )

    def render_instance_seg_pass(self, output_path: Path) -> None:
        original_pass_indices = self.snapshot_object_pass_indices()
        self.assign_instance_ids()
        temp_output_path = self.get_instance_seg_temp_path(output_path)

        def finalize_instance_seg_output(instance_seg: np.ndarray) -> None:
            self.save_instance_seg_outputs(
                output_path=output_path,
                instance_seg=instance_seg,
            )

        try:
            self.render_temp_output_pass(
                output_path=output_path,
                temp_output_path=temp_output_path,
                add_output_node=self.add_instance_seg_output_node,
                load_temp_output=self.load_instance_seg_temp_output,
                finalize_output=finalize_instance_seg_output,
                color_mode="BW",
                color_depth="16",
                enable_object_index_pass=True,
            )
        finally:
            self.restore_object_pass_indices(original_pass_indices)

    def render(self, output_path: Path) -> None:
        """Run the requested render passes and write final outputs."""
        self.scene.use_nodes = False
        auxiliary_outputs: list[tuple[Path, Path]] = []
        needs_base_render = bool({"rgb", "depth"} & set(self.render_passes))

        if "depth" in self.render_passes:
            auxiliary_outputs = self.configure_auxiliary_outputs(
                output_path, self.render_passes, self.depth_mode
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if "rgb" in self.render_passes:
            self.scene.render.filepath = str(output_path)

        if needs_base_render:
            bpy.ops.render.render(write_still="rgb" in self.render_passes)

        for temp_path, final_path in auxiliary_outputs:
            if final_path == self.get_depth_vis_output_path(output_path):
                self.finalize_depth_output(temp_path, final_path)
                continue
            raise ValueError(f"Unsupported render output target: {final_path}")
        if auxiliary_outputs:
            self.clear_compositor_tree()
            self.scene.use_nodes = False

        if "normal" in self.render_passes:
            self.render_normal_pass(output_path)
        if "mesh" in self.render_passes:
            self.render_mesh_pass(output_path)
        if "instance_seg" in self.render_passes:
            self.render_instance_seg_pass(output_path)
        if "flow" in self.render_passes:
            self.render_flow_pass(output_path)
        self.replicate_duplicate_preview_outputs(output_path)
        self.save_composite_preview(output_path)

    def run(self) -> None:
        """Prepare the scene, configure rendering, and execute all passes."""
        rgb_output_path = self.get_rgb_output_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clear_scene()
        self.import_usd()
        if self.remove_doors:
            self.remove_door_objects()
        self.validate_glb_args()
        imported_glb_objects = self.import_glb_asset()
        self.place_glb_asset(imported_glb_objects)
        min_corner, max_corner = self.get_scene_bbox()
        center = (min_corner + max_corner) * 0.5
        diagonal = (max_corner - min_corner).length
        self.create_camera()
        self.ensure_lighting(diagonal, center, max_corner.z)
        world_created = self.ensure_world()
        self.add_fill_light(
            diagonal,
            center,
            max_corner.z,
            energy=self.fill_light_energy,
        )
        if world_created:
            self.add_light_rig(
                diagonal,
                center,
                max_corner.z,
                area_energy=1500.0,
                sun_energy=0.35,
                prefix="Fill",
            )
        self.configure_color_management()
        self.configure_cycles()
        with tempfile.TemporaryDirectory(
            prefix="render_usd_", dir=None
        ) as temp_dir:
            self.temp_dir = Path(temp_dir)
            self.render(rgb_output_path)
            self.temp_dir = None

        logger.info("Rendered outputs to %s", self.output_dir)

    def compute_trajectory_frame_transforms(
        self,
        x: float,
        y: float,
        rot_deg: float,
        *,
        camera_z: float,
        camera_pitch: float,
        camera_roll: float,
        glb_offset_xyz: tuple[float, float, float],
        glb_base_rotation_deg: tuple[float, float, float],
    ) -> tuple[Vector, Euler, Matrix]:
        """Compute per-frame camera and GLB transforms for one waypoint.

        The camera sits at (x, y, camera_z) and yaws by ``rot_deg`` (its
        heading). The GLB keeps a fixed offset relative to the camera: the
        horizontal offset rotates with the yaw and its yaw adds ``rot_deg``,
        while its height stays on the floor.

        Args:
            x: Waypoint X (m).
            y: Waypoint Y (m).
            rot_deg: Waypoint heading in degrees (camera/GLB yaw).
            camera_z: Camera height (m).
            camera_pitch: Camera pitch deg (rotation_deg[0]).
            camera_roll: Camera roll deg (rotation_deg[1]).
            glb_offset_xyz: GLB-minus-camera offset at yaw=0 (world frame).
            glb_base_rotation_deg: GLB rotation deg at yaw=0.

        Returns:
            Tuple of (camera_location, camera_euler, glb_world_transform).
        """
        camera_location = Vector((x, y, camera_z))
        camera_euler = Euler(
            self.get_rotation_radians((camera_pitch, camera_roll, rot_deg)),
            "XYZ",
        )

        yaw = math.radians(rot_deg)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        off_x, off_y, off_z = glb_offset_xyz
        glb_xyz = (
            x + off_x * cos_yaw - off_y * sin_yaw,
            y + off_x * sin_yaw + off_y * cos_yaw,
            camera_z + off_z,
        )
        base_rx, base_ry, base_rz = glb_base_rotation_deg
        glb_rotation_deg = (base_rx + 90.0, base_ry, base_rz + rot_deg)
        glb_transform = self.build_camera_matrix_world(
            glb_xyz, glb_rotation_deg
        )
        return camera_location, camera_euler, glb_transform

    def compute_camera_follow_glb_transform(
        self,
        camera_xyz: tuple[float, float, float],
        camera_rotation_deg: tuple[float, float, float],
        *,
        glb_offset_xyz: tuple[float, float, float],
        glb_base_rotation_deg: tuple[float, float, float],
    ) -> Matrix:
        """Compute a GLB transform from the default camera-follow offset."""
        _, _, glb_transform = self.compute_trajectory_frame_transforms(
            camera_xyz[0],
            camera_xyz[1],
            camera_rotation_deg[2],
            camera_z=camera_xyz[2],
            camera_pitch=camera_rotation_deg[0],
            camera_roll=camera_rotation_deg[1],
            glb_offset_xyz=glb_offset_xyz,
            glb_base_rotation_deg=glb_base_rotation_deg,
        )
        return glb_transform

    def frame_outputs_exist(self) -> bool:
        """Whether every requested pass output for the current frame exists.

        Used for per-frame resume: a frame is skipped only when all its
        requested pass preview outputs are already on disk.
        """
        rgb_path = self.get_rgb_output_path()
        for pass_name, occurrence in self.iter_render_pass_occurrences():
            preview = self.get_preview_output_path(
                rgb_path, pass_name, occurrence
            )
            if preview is None or not preview.exists():
                return False
        return True

    def run_trajectory(
        self,
        *,
        waypoints: list[tuple[float, float, float]],
        camera_z: float,
        camera_pitch: float,
        camera_roll: float,
        glb_offset_xyz: tuple[float, float, float],
        glb_base_rotation_deg: tuple[float, float, float],
        frame_stride: int,
        frame_indices: list[int] | None = None,
        make_video: bool = False,
        video_fps: float = 10.0,
        overwrite: bool = False,
    ) -> None:
        """Render one frame per trajectory waypoint following the camera path.

        The scene (USD, GLB, lighting, world) is set up once; each frame only
        repositions the camera and GLB, then renders into a shared ``frames/``
        directory with a ``frame_<idx>_`` filename prefix. When ``make_video``
        is set, the RGB frames are stitched into an mp4.

        Args:
            waypoints: List of (x, y, rot_deg) waypoints.
            camera_z: Camera height (m).
            camera_pitch: Camera pitch deg (rotation_deg[0]).
            camera_roll: Camera roll deg (rotation_deg[1]).
            glb_offset_xyz: GLB-minus-camera offset at yaw=0 (world frame).
            glb_base_rotation_deg: GLB rotation deg at yaw=0.
            frame_stride: Render every Nth waypoint.
            frame_indices: Optional zero-based waypoint indices to render.
            make_video: Stitch the RGB frames into an mp4 after rendering.
            video_fps: Frame rate of the stitched video.
            overwrite: Re-render every frame; if False, frames whose pass
                outputs already exist are skipped (per-frame resume).
        """
        if not waypoints:
            raise ValueError("Trajectory contains no waypoints.")
        if self.glb_path is None:
            raise ValueError(
                "--glb_path is required for trajectory rendering."
            )
        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {self.glb_path}")
        if self.glb_path.suffix.lower() != ".glb":
            raise ValueError(
                f"Expected a .glb asset, but got: {self.glb_path}"
            )

        base_output_dir = self.output_dir
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self.clear_scene()
        self.import_usd()
        if self.remove_doors:
            self.remove_door_objects()
        imported_glb_objects = self.import_glb_asset()
        root_objects = self.get_imported_root_objects(imported_glb_objects)
        original_matrices = {
            obj.as_pointer(): obj.matrix_world.copy() for obj in root_objects
        }

        min_corner, max_corner = self.get_scene_bbox()
        center = (min_corner + max_corner) * 0.5
        diagonal = (max_corner - min_corner).length
        camera = self.create_camera()
        self.ensure_lighting(diagonal, center, max_corner.z)
        world_created = self.ensure_world()
        self.add_fill_light(
            diagonal, center, max_corner.z, energy=self.fill_light_energy
        )
        if world_created:
            self.add_light_rig(
                diagonal,
                center,
                max_corner.z,
                area_energy=1500.0,
                sun_energy=0.35,
                prefix="Fill",
            )
        self.configure_color_management()
        self.configure_cycles()

        if frame_indices is None:
            frames = list(enumerate(waypoints))[:: max(1, frame_stride)]
        else:
            invalid_indices = sorted(
                {
                    index
                    for index in frame_indices
                    if index < 0 or index >= len(waypoints)
                }
            )
            if invalid_indices:
                raise ValueError(
                    "Trajectory frame indices out of range "
                    f"[0, {len(waypoints) - 1}]: {invalid_indices}"
                )
            frames = [
                (index, waypoints[index])
                for index in dict.fromkeys(frame_indices)
            ]
        self.frame_layout = True
        self.frame_base_dir = base_output_dir

        # The highest already-rendered frame is the one most likely left
        # half-written by an interrupted run, so re-render it even on resume.
        last_existing = -1
        if not overwrite:
            for frame_idx, _ in frames:
                self.frame_index = frame_idx
                if self.frame_outputs_exist():
                    last_existing = frame_idx

        with tempfile.TemporaryDirectory(
            prefix="render_usd_traj_", dir=None
        ) as temp_dir:
            self.temp_dir = Path(temp_dir)
            for frame_idx, (x, y, rot_deg) in frames:
                self.frame_index = frame_idx
                # Per-frame resume: skip complete frames, but always redo the
                # last existing one (possibly a truncated/corrupt write).
                if (
                    not overwrite
                    and frame_idx != last_existing
                    and self.frame_outputs_exist()
                ):
                    logger.info("Skip existing frame %04d", frame_idx)
                    continue

                cam_loc, cam_euler, glb_transform = (
                    self.compute_trajectory_frame_transforms(
                        x,
                        y,
                        rot_deg,
                        camera_z=camera_z,
                        camera_pitch=camera_pitch,
                        camera_roll=camera_roll,
                        glb_offset_xyz=glb_offset_xyz,
                        glb_base_rotation_deg=glb_base_rotation_deg,
                    )
                )
                camera.location = cam_loc
                camera.rotation_euler = cam_euler
                for obj in root_objects:
                    obj.matrix_world = (
                        glb_transform @ original_matrices[obj.as_pointer()]
                    )
                bpy.context.view_layer.update()

                self.render(self.get_rgb_output_path())
                logger.info(
                    "Rendered frame %04d at (%.3f, %.3f, rot=%.1f)",
                    frame_idx,
                    x,
                    y,
                    rot_deg,
                )
            self.temp_dir = None

        self.frame_layout = False
        self.frame_base_dir = None
        self.output_dir = base_output_dir
        logger.info(
            "Rendered %d trajectory frames to %s",
            len(frames),
            base_output_dir,
        )

        if make_video:
            self.stitch_pass_videos(base_output_dir, fps=video_fps)

    def run_camera_trajectory(
        self,
        *,
        frames: list[CameraTrajectoryFrame],
        glb_offset_xyz: tuple[float, float, float],
        glb_base_rotation_deg: tuple[float, float, float],
        frame_stride: int,
        frame_indices: list[int] | None = None,
        make_video: bool = False,
        video_fps: float = 10.0,
        overwrite: bool = False,
    ) -> None:
        """Render fully specified camera poses loaded from a trajectory CSV.

        Args:
            frames: Camera frames with output index, timestamp, and 6DoF pose.
            glb_offset_xyz: Local XYZ offset applied to GLB assets.
            glb_base_rotation_deg: Base XYZ Euler rotation applied to GLB assets.
            frame_stride: Render every Nth CSV row.
            frame_indices: Optional CSV ``frame`` values to render.
            make_video: Stitch rendered image passes into mp4 files.
            video_fps: Frame rate of the stitched videos.
            overwrite: Re-render complete frames instead of resuming.
        """
        if not frames:
            raise ValueError("Camera trajectory contains no frames.")

        base_output_dir = self.output_dir
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self.clear_scene()
        self.import_usd()
        if self.remove_doors:
            self.remove_door_objects()
        self.validate_glb_args(allow_path_only=True)
        imported_glb_objects = self.import_glb_asset()
        glb_follows_camera = (
            self.glb_path is not None
            and self.glb_xyz is None
            and self.glb_rotation_deg is None
        )
        root_objects: list[bpy.types.Object] = []
        original_matrices: dict[int, Matrix] = {}
        if glb_follows_camera:
            root_objects = self.get_imported_root_objects(imported_glb_objects)
            original_matrices = {
                obj.as_pointer(): obj.matrix_world.copy()
                for obj in root_objects
            }
        else:
            self.place_glb_asset(imported_glb_objects)

        min_corner, max_corner = self.get_scene_bbox()
        center = (min_corner + max_corner) * 0.5
        diagonal = (max_corner - min_corner).length
        camera = self.create_camera()
        self.ensure_lighting(diagonal, center, max_corner.z)
        world_created = self.ensure_world()
        self.add_fill_light(
            diagonal, center, max_corner.z, energy=self.fill_light_energy
        )
        if world_created:
            self.add_light_rig(
                diagonal,
                center,
                max_corner.z,
                area_energy=1500.0,
                sun_energy=0.35,
                prefix="Fill",
            )
        self.configure_color_management()
        self.configure_cycles()

        if frame_indices is None:
            selected_frames = frames[:: max(1, frame_stride)]
        else:
            frames_by_index = {frame.frame_index: frame for frame in frames}
            invalid_indices = sorted(
                {
                    index
                    for index in frame_indices
                    if index not in frames_by_index
                }
            )
            if invalid_indices:
                raise ValueError(
                    "Camera trajectory frame indices not found: "
                    f"{invalid_indices}"
                )
            selected_frames = [
                frames_by_index[index]
                for index in dict.fromkeys(frame_indices)
            ]

        self.frame_layout = True
        self.frame_base_dir = base_output_dir

        last_existing = -1
        if not overwrite:
            for frame in selected_frames:
                self.frame_index = frame.frame_index
                if self.frame_outputs_exist():
                    last_existing = frame.frame_index

        with tempfile.TemporaryDirectory(
            prefix="render_usd_camera_traj_", dir=None
        ) as temp_dir:
            self.temp_dir = Path(temp_dir)
            for frame in selected_frames:
                self.frame_index = frame.frame_index
                if (
                    not overwrite
                    and frame.frame_index != last_existing
                    and self.frame_outputs_exist()
                ):
                    logger.info("Skip existing frame %04d", frame.frame_index)
                    continue

                camera.location = Vector(frame.xyz)
                camera.rotation_euler = Euler(
                    self.get_rotation_radians(frame.rotation_deg),
                    "XYZ",
                )
                if glb_follows_camera:
                    glb_transform = self.compute_camera_follow_glb_transform(
                        frame.xyz,
                        frame.rotation_deg,
                        glb_offset_xyz=glb_offset_xyz,
                        glb_base_rotation_deg=glb_base_rotation_deg,
                    )
                    for obj in root_objects:
                        obj.matrix_world = (
                            glb_transform @ original_matrices[obj.as_pointer()]
                        )
                bpy.context.view_layer.update()

                self.render(self.get_rgb_output_path())
                logger.info(
                    "Rendered frame %04d at "
                    "(%.3f, %.3f, %.3f), rotation=(%.2f, %.2f, %.2f)",
                    frame.frame_index,
                    *frame.xyz,
                    *frame.rotation_deg,
                )
            self.temp_dir = None

        self.frame_layout = False
        self.frame_base_dir = None
        self.output_dir = base_output_dir
        logger.info(
            "Rendered %d camera trajectory frames to %s",
            len(selected_frames),
            base_output_dir,
        )

        if make_video:
            self.stitch_pass_videos(base_output_dir, fps=video_fps)

    def stitch_pass_videos(self, base_dir: Path, *, fps: float) -> None:
        """Stitch each ``frames_<pass>/`` image sequence into ``<pass>.mp4``.

        Scans ``base_dir`` for ``frames_<pass>`` directories, orders the
        ``frame_*.png`` images, and writes one video per pass next to them
        (e.g. ``frames_render_rgb/`` -> ``render_rgb.mp4``) using ffmpeg.

        Args:
            base_dir: Trajectory output directory containing the frame dirs.
            fps: Output frame rate (clamped to a small positive minimum).
        """
        fps = max(fps, 1.0)
        for frames_subdir in sorted(base_dir.glob("frames_*")):
            if not frames_subdir.is_dir():
                continue
            frame_paths = sorted(frames_subdir.glob("frame_*.png"))
            if not frame_paths:
                continue

            pass_name = frames_subdir.name[len("frames_") :]
            output_path = base_dir / f"{pass_name}.mp4"
            self.encode_frames_to_video(frames_subdir, output_path, fps)
            logger.info(
                "Stitched %d frames into %s (%.2f fps).",
                len(frame_paths),
                output_path,
                fps,
            )

    @staticmethod
    def encode_frames_to_video(
        frames_dir: Path, output_path: Path, fps: float
    ) -> None:
        """Encode a ``frame_*.png`` sequence into an mp4 via ffmpeg.

        Uses the system ffmpeg (h264/yuv420p) and a glob input pattern so
        non-contiguous frame indices (e.g. with --frame_stride) work. Even
        dimensions are enforced for yuv420p compatibility.

        Args:
            frames_dir: Directory holding the ``frame_*.png`` images.
            output_path: Destination mp4 path.
            fps: Output frame rate.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-pattern_type",
            "glob",
            "-i",
            str(frames_dir / "frame_*.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {output_path}:\n{result.stderr}"
            )


def load_trajectory_waypoints(
    trajectory_json: Path,
) -> list[tuple[float, float, float]]:
    """Load (x, y, rot) waypoints from a trajectory JSON file.

    Accepts either a dict with a ``waypoints`` key or a bare list of records.

    Args:
        trajectory_json: Path to the trajectory JSON.

    Returns:
        List of (x, y, rot_deg) tuples.
    """
    data = json.loads(Path(trajectory_json).read_text())
    records = data["waypoints"] if isinstance(data, dict) else data
    return [
        (float(rec["x"]), float(rec["y"]), float(rec["rot"]))
        for rec in records
    ]


def load_camera_trajectory_csv(
    trajectory_csv: Path,
) -> list[CameraTrajectoryFrame]:
    """Load full camera poses from a trajectory CSV file.

    Args:
        trajectory_csv: CSV containing the required camera trajectory fields.

    Returns:
        Camera frames in CSV row order.
    """
    trajectory_csv = Path(trajectory_csv)
    if not trajectory_csv.exists():
        raise FileNotFoundError(
            f"Camera trajectory CSV not found: {trajectory_csv}"
        )

    frames: list[CameraTrajectoryFrame] = []
    seen_indices: set[int] = set()
    previous_frame_index: int | None = None
    previous_time_sec: float | None = None
    with trajectory_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or ())
        missing_fields = [
            field
            for field in CAMERA_TRAJECTORY_CSV_FIELDS
            if field not in fieldnames
        ]
        if missing_fields:
            raise ValueError(
                f"Camera trajectory CSV is missing fields: {missing_fields}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                frame_index = int(row["frame"])
                values = {
                    field: float(row[field])
                    for field in CAMERA_TRAJECTORY_CSV_FIELDS[1:]
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid camera trajectory value at CSV row "
                    f"{row_number}: {exc}"
                ) from exc

            if frame_index < 0:
                raise ValueError(
                    f"Camera frame must be non-negative at CSV row "
                    f"{row_number}: {frame_index}"
                )
            if frame_index in seen_indices:
                raise ValueError(
                    f"Duplicate camera frame at CSV row {row_number}: "
                    f"{frame_index}"
                )
            non_finite_fields = [
                field
                for field, value in values.items()
                if not math.isfinite(value)
            ]
            if non_finite_fields:
                raise ValueError(
                    f"Non-finite values at CSV row {row_number}: "
                    f"{non_finite_fields}"
                )
            if (
                previous_frame_index is not None
                and frame_index <= previous_frame_index
            ):
                raise ValueError(
                    "Camera frames must be strictly increasing at CSV row "
                    f"{row_number}: {frame_index} <= {previous_frame_index}"
                )
            if (
                previous_time_sec is not None
                and values["time_sec"] <= previous_time_sec
            ):
                raise ValueError(
                    "Camera timestamps must be strictly increasing at CSV "
                    f"row {row_number}: {values['time_sec']} <= "
                    f"{previous_time_sec}"
                )

            seen_indices.add(frame_index)
            previous_frame_index = frame_index
            previous_time_sec = values["time_sec"]
            frames.append(
                CameraTrajectoryFrame(
                    frame_index=frame_index,
                    time_sec=values["time_sec"],
                    xyz=(values["x"], values["y"], values["z"]),
                    rotation_deg=(
                        values["roll_x_deg"],
                        values["pitch_y_deg"],
                        values["yaw_z_deg"],
                    ),
                )
            )

    return frames


def resolve_video_fps(
    video_fps: float, trajectory_json: Path, frame_stride: int
) -> float:
    """Resolve the video frame rate, auto-deriving real-time fps if 0.

    Args:
        video_fps: Requested fps; 0 means auto.
        trajectory_json: Trajectory JSON (read for ``time_step``).
        frame_stride: Waypoint stride used during rendering.

    Returns:
        A positive frame rate.
    """
    if video_fps > 0.0:
        return video_fps

    data = json.loads(Path(trajectory_json).read_text())
    time_step = data.get("time_step") if isinstance(data, dict) else None
    if not time_step or time_step <= 0.0:
        return 10.0
    return (1.0 / time_step) / max(1, frame_stride)


def resolve_csv_video_fps(
    video_fps: float,
    frames: list[CameraTrajectoryFrame],
    frame_stride: int,
) -> float:
    """Resolve video fps from CSV timestamps when no fps is specified."""
    if video_fps > 0.0:
        return video_fps

    time_deltas = [
        current.time_sec - previous.time_sec
        for previous, current in zip(frames, frames[1:])
        if current.time_sec > previous.time_sec
    ]
    if not time_deltas:
        return 10.0

    time_step = float(np.median(time_deltas))
    return (1.0 / time_step) / max(1, frame_stride)


def run_trajectory_render(args: argparse.Namespace) -> None:
    """Render a trajectory from parsed CLI arguments."""
    waypoints = load_trajectory_waypoints(args.trajectory_json)
    if not waypoints:
        raise ValueError(f"No waypoints found in {args.trajectory_json}.")

    first_x, first_y, first_rot = waypoints[0]
    renderer = RenderUsd(
        usd_path=args.usd_path,
        glb_path=args.glb_path,
        glb_xyz=None,
        glb_rotation_deg=None,
        output_dir=args.output_dir,
        render_passes=args.render_passes,
        depth_mode=args.depth_mode,
        depth_max=args.depth_max,
        resolution=args.resolution,
        samples=args.samples,
        camera_xyz=(first_x, first_y, args.camera_z),
        camera_rotation_deg=(args.camera_pitch, args.camera_roll, first_rot),
        flow_camera_xyz=args.flow_camera_xyz,
        flow_camera_rotation_deg=args.flow_camera_rotation_deg,
        focal_length_mm=args.focal_length_mm,
        exposure=args.exposure,
        world_strength=args.world_strength,
        fill_light_energy=args.fill_light_energy,
        remove_doors=args.remove_doors,
    )
    renderer.run_trajectory(
        waypoints=waypoints,
        camera_z=args.camera_z,
        camera_pitch=args.camera_pitch,
        camera_roll=args.camera_roll,
        glb_offset_xyz=tuple(args.glb_offset_xyz),
        glb_base_rotation_deg=tuple(args.glb_base_rotation_deg),
        frame_stride=args.frame_stride,
        frame_indices=args.frame_indices,
        make_video=args.make_video,
        video_fps=resolve_video_fps(
            args.video_fps, args.trajectory_json, args.frame_stride
        ),
        overwrite=args.overwrite,
    )


def run_camera_trajectory_render(args: argparse.Namespace) -> None:
    """Render a full-pose camera trajectory from parsed CLI arguments."""
    frames = load_camera_trajectory_csv(args.trajectory_csv)
    if not frames:
        raise ValueError(f"No frames found in {args.trajectory_csv}.")

    first_frame = frames[0]
    renderer = RenderUsd(
        usd_path=args.usd_path,
        glb_path=args.glb_path,
        glb_xyz=args.glb_xyz,
        glb_rotation_deg=args.glb_rotation_deg,
        output_dir=args.output_dir,
        render_passes=args.render_passes,
        depth_mode=args.depth_mode,
        depth_max=args.depth_max,
        resolution=args.resolution,
        samples=args.samples,
        camera_xyz=first_frame.xyz,
        camera_rotation_deg=first_frame.rotation_deg,
        flow_camera_xyz=args.flow_camera_xyz,
        flow_camera_rotation_deg=args.flow_camera_rotation_deg,
        focal_length_mm=args.focal_length_mm,
        exposure=args.exposure,
        world_strength=args.world_strength,
        fill_light_energy=args.fill_light_energy,
        remove_doors=args.remove_doors,
    )
    renderer.run_camera_trajectory(
        frames=frames,
        glb_offset_xyz=tuple(args.glb_offset_xyz),
        glb_base_rotation_deg=tuple(args.glb_base_rotation_deg),
        frame_stride=args.frame_stride,
        frame_indices=args.frame_indices,
        make_video=args.make_video,
        video_fps=resolve_csv_video_fps(
            args.video_fps, frames, args.frame_stride
        ),
        overwrite=args.overwrite,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.trajectory_json is not None:
        run_trajectory_render(args)
        return
    if args.trajectory_csv is not None:
        run_camera_trajectory_render(args)
        return

    if args.camera_xyz is None or args.camera_rotation_deg is None:
        build_arg_parser().error(
            "--camera_xyz and --camera_rotation_deg are required unless "
            "--trajectory_json or --trajectory_csv is given."
        )
    RenderUsd.from_args(args).run()


if __name__ == "__main__":
    main()
