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

import json
import logging
import subprocess
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import tyro
from shapely.geometry import Polygon
from shapely.ops import unary_union

# The spatial-computing package name contains a hyphen, so it cannot be
# imported with a normal ``import`` statement; use importlib instead.
_SC = "embodied_gen.skills.spatial-computing.core"
UrdfSemanticInfoCollector = import_module(
    f"{_SC}.collector"
).UrdfSemanticInfoCollector
RoamTrajectoryGenerator = import_module(
    f"{_SC}.trajectory"
).RoamTrajectoryGenerator
FloorplanVisualizer = import_module(f"{_SC}.visualizer").FloorplanVisualizer

logger = logging.getLogger(__name__)


@dataclass
class GenTrajectoryArgs:
    """Configuration for roaming-trajectory generation."""

    urdf_path: str
    """Path to the input URDF scene file."""

    output_dir: str = "outputs/trajectory"
    """Directory for the trajectory JSON and floorplan PNG."""

    clearance: float = 0.4
    """Minimum clearance (m) from walls; keep small so doorways stay passable."""

    obstacle_clearance: float = 0.5
    """Minimum clearance (m) from furniture/objects (doors/walls excluded)."""

    resolution: float = 0.05
    """Occupancy-grid cell size in meters (smaller = finer, slower)."""

    num_waypoints: int = 8
    """Number of roam waypoints to visit across the navigable space."""

    speed: float = 0.5
    """Constant roaming speed (m/s). Sets the spacing with --time_step."""

    time_step: float = 0.1
    """Time between consecutive waypoints (s). Spacing = speed * time_step."""

    turn_radius: float = 0.8
    """Target turning-arc radius (m); larger gives wider, rounder turns."""

    endpoint_clearance: float = 1.5
    """Minimum distance (m) the start and end keep from walls/obstacles."""

    arrow_stride: int = 10
    """Draw a heading arrow every N points in the PNG (0 disables)."""

    remove_doors: bool = False
    """Hide door footprints in the floorplan image and animation."""

    animate: bool = False
    """Also render a per-frame floorplan animation (frames/ + trajectory.mp4)."""

    anim_frame_stride: int = 1
    """Animate every Nth waypoint. Match the render --frame_stride to sync."""

    anim_fps: float = 0.0
    """Animation frame rate. 0 = auto (real-time from speed / stride)."""

    anim_dpi: int = 100
    """Resolution (dpi) of the animation frames."""

    start_xy: tuple[float, float] | None = None
    """Optional world (x, y) start; defaults to the largest free region."""

    mesh_sample_num: int = 5000
    """Number of points sampled per mesh when parsing the URDF."""

    seed: int | None = None
    """Optional RNG seed for reproducible roaming."""


def generate_trajectory(cfg: GenTrajectoryArgs) -> None:
    """Run the roaming-trajectory pipeline and write outputs.

    Args:
        cfg: Parsed CLI configuration.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = UrdfSemanticInfoCollector(mesh_sample_num=cfg.mesh_sample_num)
    collector.collect(cfg.urdf_path)

    generator = RoamTrajectoryGenerator.from_collector(
        collector,
        clearance=cfg.clearance,
        resolution=cfg.resolution,
        obstacle_clearance=cfg.obstacle_clearance,
    )
    result = generator.generate(
        start_xy=cfg.start_xy,
        num_waypoints=cfg.num_waypoints,
        point_spacing=cfg.speed * cfg.time_step,
        turn_radius=cfg.turn_radius,
        endpoint_clearance=cfg.endpoint_clearance,
        seed=cfg.seed,
    )

    stem = Path(cfg.urdf_path).stem
    json_path = output_dir / f"{stem}_trajectory.json"
    png_path = output_dir / f"{stem}_trajectory.png"

    # Points are equidistant, so timestamps follow constant-speed motion:
    # t = cumulative arc length / speed.
    xy = result.points[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    times = np.concatenate([[0.0], np.cumsum(seg)]) / cfg.speed
    waypoints = [
        {
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "rot": round(float(r), 2),
            "t": round(float(t), 3),
        }
        for (x, y, r), t in zip(result.points, times)
    ]

    payload = {
        "num_waypoints": len(waypoints),
        "clearance": result.clearance,
        "min_clearance": round(result.min_clearance, 4),
        "length": round(result.length, 4),
        "speed": cfg.speed,
        "time_step": cfg.time_step,
        "reachable_rooms": result.reachable_rooms,
        "rot_convention": (
            "0 deg points +Y (12 o'clock); counter-clockwise positive; "
            "tangent to the roaming curve (forward heading); range [0, 360)."
        ),
        "waypoints": waypoints,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    footprints, occ_area = visible_footprints(collector, cfg.remove_doors)

    FloorplanVisualizer.plot(
        collector.rooms,
        footprints,
        occ_area,
        str(png_path),
        trajectory=result.points,
        arrow_stride=cfg.arrow_stride,
    )

    logger.info("Trajectory waypoints : %s", json_path)
    logger.info("Floorplan overlay    : %s", png_path)
    logger.info(
        "Length %.2fm, min clearance %.2fm, reaches rooms: %s",
        result.length,
        result.min_clearance,
        ", ".join(result.reachable_rooms) or "(none)",
    )

    if cfg.animate:
        fps = cfg.anim_fps
        if fps <= 0.0:
            fps = (1.0 / cfg.time_step) / max(1, cfg.anim_frame_stride)
        animate_floorplan(
            rooms=collector.rooms,
            footprints=footprints,
            occ_area=occ_area,
            points=result.points,
            output_dir=output_dir,
            frame_stride=cfg.anim_frame_stride,
            fps=fps,
            dpi=cfg.anim_dpi,
        )


def visible_footprints(collector, remove_doors: bool):
    """Return footprints and occupied area for floorplan drawing.

    When ``remove_doors`` is set, door footprints are excluded and the
    occupied area is recomputed without them so doors do not appear.

    Args:
        collector: Parsed scene collector.
        remove_doors: Whether to hide door footprints.

    Returns:
        Tuple of (footprints dict, occupied-area geometry).
    """
    if not remove_doors:
        return collector.footprints, collector.occ_area

    footprints = {
        key: poly
        for key, poly in collector.footprints.items()
        if "door" not in key.lower()
    }
    occ_area = (
        unary_union(list(footprints.values())) if footprints else Polygon()
    )
    return footprints, occ_area


def animate_floorplan(
    rooms: dict,
    footprints: dict,
    occ_area,
    points: np.ndarray,
    output_dir: Path,
    frame_stride: int,
    fps: float,
    dpi: int,
) -> None:
    """Render a floorplan animation that follows the trajectory progress.

    Each frame shows the traveled path in red, a green dot at the current
    position, and a red heading arrow; the future path is hidden. Frame
    indices match the render frames (same stride) for easy pairing.

    Args:
        rooms: Room-name to floor-polygon map.
        footprints: Object-name to footprint-polygon map (to draw).
        occ_area: Occupied-area geometry overlay.
        points: (N, 3) array of (x, y, rot_deg) waypoints.
        output_dir: Directory to write ``frames/`` and ``trajectory.mp4``.
        frame_stride: Render every Nth waypoint.
        fps: Output video frame rate.
        dpi: Resolution of each frame.
    """
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(points), max(1, frame_stride)):
        FloorplanVisualizer.plot(
            rooms,
            footprints,
            occ_area,
            str(frames_dir / f"frame_{i:04d}.png"),
            trajectory=points,
            current_index=i,
            dpi=dpi,
        )

    video_path = output_dir / "trajectory.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{max(fps, 1.0):g}",
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
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}:\n{result.stderr}")

    logger.info("Trajectory animation : %s", video_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = tyro.cli(GenTrajectoryArgs)
    generate_trajectory(cfg)


if __name__ == "__main__":
    main()
