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


"""Render a top-down bird's-eye view of a USD scene with ceilings hidden."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector
from embodied_gen.scripts.room_gen.render_usd import RenderUsd

logger = logging.getLogger(__name__)


CEILING_KEYWORDS = ("ceiling", "exterior")
DEFAULT_ROOM_USD_GLOB = "seed*/usd/export_scene/export_scene.usdc"


class BirdseyeRenderUsd(RenderUsd):
    """Top-down USD renderer with ceiling removal and orthographic camera."""

    def __init__(
        self,
        *,
        ortho_margin: float = 1.05,
        use_cpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ortho_margin = ortho_margin
        self.use_cpu = use_cpu

    def configure_cycles(self) -> None:
        if self.use_cpu:
            self.scene.render.engine = "CYCLES"
            self.scene.cycles.device = "CPU"
            self.scene.cycles.samples = self.samples
            self.scene.render.resolution_x = self.resolution[0]
            self.scene.render.resolution_y = self.resolution[1]
            self.scene.render.image_settings.file_format = "PNG"
        else:
            super().configure_cycles()
        self.scene.render.film_transparent = True
        self.scene.render.image_settings.color_mode = "RGBA"

    def remove_ceiling_objects(self) -> int:
        """Delete any object whose name contains a ceiling keyword."""
        removed = 0
        for obj in list(self.scene.objects):
            lower = obj.name.lower()
            if any(kw in lower for kw in CEILING_KEYWORDS):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
        logger.info("Removed %d ceiling objects.", removed)
        return removed

    def create_orthographic_camera(
        self, center: Vector, top_z: float, scene_size: float
    ) -> bpy.types.Object:
        location = Vector((center.x, center.y, top_z + max(scene_size, 1.0)))
        bpy.ops.object.camera_add(location=location, rotation=(0.0, 0.0, 0.0))
        camera = bpy.context.object
        camera.rotation_mode = "XYZ"
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = scene_size * self.ortho_margin
        camera.data.clip_start = 0.01
        camera.data.clip_end = 1000.0
        self.scene.camera = camera
        return camera

    def run(self) -> None:
        rgb_output_path = self.get_rgb_output_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clear_scene()
        self.import_usd()
        self.remove_ceiling_objects()
        self.validate_glb_args()
        imported_glb_objects = self.import_glb_asset()
        self.place_glb_asset(imported_glb_objects)

        min_corner, max_corner = self.get_scene_bbox()
        center = (min_corner + max_corner) * 0.5
        diagonal = (max_corner - min_corner).length
        scene_size = max(
            max_corner.x - min_corner.x, max_corner.y - min_corner.y
        )

        self.create_orthographic_camera(center, max_corner.z, scene_size)
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
            prefix="render_birdseye_", dir=None
        ) as temp_dir:
            self.temp_dir = Path(temp_dir)
            self.render(rgb_output_path)
            self.temp_dir = None

        logger.info("Rendered bird's-eye outputs to %s", self.output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a top-down bird's-eye view of a USD scene."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--usd_path", type=Path)
    input_group.add_argument(
        "--input_dir",
        type=Path,
        help=(
            "Directory with seed*/usd/export_scene/export_scene.usdc files "
            "to render in batch."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for a single --usd_path render.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        help=(
            "Batch output root. Defaults to <input_dir>/bev, with one "
            "subdirectory per seed."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip batch items that already have render_rgb.png.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1920),
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--exposure", type=float, default=-1.0)
    parser.add_argument("--world_strength", type=float, default=1.0)
    parser.add_argument("--fill_light_energy", type=float, default=1000.0)
    parser.add_argument("--ortho_margin", type=float, default=1.05)
    parser.add_argument("--use_cpu", action="store_true")
    return parser


def find_room_usd_paths(input_dir: Path) -> list[Path]:
    """Find seed room USD files under an input directory."""
    return sorted(input_dir.glob(DEFAULT_ROOM_USD_GLOB))


def get_batch_output_dir(
    usd_path: Path, input_dir: Path, output_root: Path
) -> Path:
    """Build the batch render output directory for a seed USD path."""
    try:
        seed_dir = usd_path.relative_to(input_dir).parts[0]
    except ValueError:
        seed_dir = usd_path.parents[2].name
    return output_root / seed_dir


def build_renderer(
    args: argparse.Namespace, usd_path: Path, output_dir: Path
) -> BirdseyeRenderUsd:
    """Build a bird's-eye renderer with shared CLI options."""
    return BirdseyeRenderUsd(
        usd_path=usd_path,
        glb_path=None,
        glb_xyz=None,
        glb_rotation_deg=None,
        output_dir=output_dir,
        render_passes=("rgb",),
        depth_mode="normalized",
        resolution=tuple(args.resolution),
        samples=args.samples,
        camera_xyz=(0.0, 0.0, 0.0),
        camera_rotation_deg=(0.0, 0.0, 0.0),
        flow_camera_xyz=None,
        flow_camera_rotation_deg=None,
        focal_length_mm=20.0,
        exposure=args.exposure,
        world_strength=args.world_strength,
        fill_light_energy=args.fill_light_energy,
        ortho_margin=args.ortho_margin,
        use_cpu=args.use_cpu,
    )


def render_single(
    args: argparse.Namespace, usd_path: Path, output_dir: Path
) -> None:
    build_renderer(args, usd_path, output_dir).run()


def render_batch(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    output_root = args.output_root or input_dir / "bev"
    usd_paths = find_room_usd_paths(input_dir)
    if not usd_paths:
        raise FileNotFoundError(
            f"No USD files found under {input_dir} matching "
            f"{DEFAULT_ROOM_USD_GLOB}."
        )

    logger.info(
        "Rendering %d bird's-eye views under %s.", len(usd_paths), input_dir
    )
    for usd_path in usd_paths:
        output_dir = get_batch_output_dir(usd_path, input_dir, output_root)
        rgb_output_path = output_dir / "render_rgb.png"
        if args.skip_existing and rgb_output_path.exists():
            logger.info("Skipping existing render %s", rgb_output_path)
            continue

        logger.info("Rendering %s to %s", usd_path, output_dir)
        render_single(args, usd_path, output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    if args.input_dir is not None:
        render_batch(args)
        return

    if args.output_dir is None:
        raise ValueError("--output_dir is required when using --usd_path.")
    render_single(args, args.usd_path, args.output_dir)


if __name__ == "__main__":
    main()
