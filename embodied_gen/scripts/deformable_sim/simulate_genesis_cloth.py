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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from tqdm import tqdm
from embodied_gen.utils.log import logger

DEFAULT_CAMERA_FOV = 45


@dataclass
class GenesisClothSimConfig:
    asset_path: str = "outputs/cloth"
    output_dir: str | None = None
    init_height: float = 1.0
    duration_seconds: float = 6.0
    sim_steps: int | None = None
    shake: bool = True
    shake_steps: int | None = None
    shake_amplitude: float = 0.08
    device: str = "cpu"
    headless: bool = True
    fps: int = 30
    render_interval: int = 1
    camera_res: tuple[int, int] = (512, 512)
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    background_color: tuple[float, float, float] = (0.78, 0.82, 0.88)
    ambient_light: tuple[float, float, float] = (0.55, 0.55, 0.55)
    floor_size: tuple[float, float] = (6.0, 6.0)
    floor_tile_size: tuple[float, float] = (0.5, 0.5)
    floor_color: tuple[float, float, float, float] = (0.72, 0.72, 0.68, 1.0)
    backdrop: bool = False
    backdrop_color: tuple[float, float, float, float] = (0.80, 0.84, 0.90, 1.0)


def simulate_genesis_cloth(
    asset_path: str = "outputs/cloth",
    output_dir: str | None = None,
    init_height: float = 1.0,
    duration_seconds: float = 6.0,
    sim_steps: int | None = None,
    shake: bool = True,
    shake_steps: int | None = None,
    shake_amplitude: float = 0.08,
    device: str = "cpu",
    headless: bool = True,
    fps: int = 30,
    render_interval: int = 1,
    camera_res: tuple[int, int] = (512, 512),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    background_color: tuple[float, float, float] = (0.78, 0.82, 0.88),
    ambient_light: tuple[float, float, float] = (0.55, 0.55, 0.55),
    floor_size: tuple[float, float] = (6.0, 6.0),
    floor_tile_size: tuple[float, float] = (0.5, 0.5),
    floor_color: tuple[float, float, float, float] = (0.72, 0.72, 0.68, 1.0),
    backdrop: bool = False,
    backdrop_color: tuple[float, float, float, float] = (
        0.80,
        0.84,
        0.90,
        1.0,
    ),
) -> str:
    """Run a Genesis cloth drop-and-shake demo.

    Args:
        asset_path: Directory containing ``genesis/manifest.json``.
        output_dir: Directory for simulation outputs. If unset, writes to
            ``asset_path/genesis_sim``.
        init_height: Height in meters for the cloth initial position.
        duration_seconds: Default output video duration when ``sim_steps`` is not
            set. Defaults to 6 seconds.
        sim_steps: Optional number of Genesis simulation steps to run. If unset,
            it is derived from ``duration_seconds * fps * render_interval``.
        shake: Whether to apply a visible nonuniform particle displacement.
        shake_steps: Number of steps that receive the shake motion. If unset,
            all simulation steps are shaken.
        shake_amplitude: Maximum displacement magnitude in meters.
        device: Genesis backend name: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        headless: Whether to run Genesis without opening the viewer.
            Genesis 0.4.7 uses Scene(show_viewer=False) for headless mode.
        fps: Output video frames per second.
        render_interval: Render every N simulation steps.
        camera_res: Output camera resolution as ``(width, height)``. Defaults
            to 512x512.
        gravity: Gravity vector. Defaults to zero for the wind-sway demo so
            the shirt does not visibly sag before the lower edge starts moving.
        background_color: RGB color for the renderer background.
        ambient_light: RGB ambient light color.
        floor_size: Visible floor plane size in meters.
        floor_tile_size: Visual tile size for the floor plane.
        floor_color: RGBA floor material color.
        backdrop: Whether to add a vertical background plane behind the cloth.
        backdrop_color: RGBA backdrop material color.

    Returns:
        Path to the generated video file.
    """

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if render_interval <= 0:
        raise ValueError("render_interval must be positive")
    if sim_steps is None:
        sim_steps = max(
            1, int(round(duration_seconds * fps * render_interval))
        )
    if sim_steps <= 0:
        raise ValueError("sim_steps must be positive")
    if shake_steps is None:
        shake_steps = sim_steps
    if shake_steps < 0:
        raise ValueError("shake_steps must be non-negative")

    asset_dir = Path(asset_path)
    manifest_path = asset_dir / "genesis" / "manifest.json"
    manifest = _read_json(manifest_path)
    material_config = _read_json(asset_dir / manifest["material"])
    cloth_mesh_path = asset_dir / manifest["cloth_mesh"]
    if not cloth_mesh_path.exists():
        raise FileNotFoundError(
            f"Converted cloth mesh not found: {cloth_mesh_path}"
        )

    sim_output_dir = (
        Path(output_dir) if output_dir else asset_dir / "genesis_sim"
    )
    sim_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Genesis cloth outputs will be saved to {sim_output_dir}")
    video_path = sim_output_dir / "video.mp4"
    summary_path = sim_output_dir / "run_summary.json"
    config_path = sim_output_dir / "sim_config.json"
    particle_positions_path = sim_output_dir / "particle_positions.npz"

    import genesis as gs

    backend_name, backend_warning = _init_genesis_backend(gs, device)

    pbd_options = gs.options.PBDOptions(
        particle_size=float(material_config.get("particle_size", 0.02)),
        gravity=tuple(float(value) for value in gravity),
    )
    camera_pos, camera_lookat = _front_camera_pose(manifest, init_height)
    scene = gs.Scene(
        show_viewer=not headless,
        sim_options=gs.options.SimOptions(
            dt=1.0 / float(fps * render_interval),
            gravity=tuple(float(value) for value in gravity),
        ),
        pbd_options=pbd_options,
        vis_options=gs.options.VisOptions(
            background_color=tuple(float(value) for value in background_color),
            ambient_light=tuple(float(value) for value in ambient_light),
            shadow=True,
            plane_reflection=False,
        ),
    )
    _add_scene_floor_and_backdrop(
        gs=gs,
        scene=scene,
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
        floor_size=floor_size,
        floor_tile_size=floor_tile_size,
        floor_color=floor_color,
        backdrop=backdrop,
        backdrop_color=backdrop_color,
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(cloth_mesh_path),
            pos=(0.0, 0.0, float(init_height)),
            scale=float(manifest.get("scale", 1.0)),
            convexify=False,
        ),
        material=gs.materials.PBD.Cloth(
            rho=float(material_config.get("rho", 0.2)),
            static_friction=float(material_config.get("static_friction", 0.5)),
            kinetic_friction=float(
                material_config.get("kinetic_friction", 0.4)
            ),
            stretch_compliance=float(
                material_config.get("stretch_compliance", 1e-6)
            ),
            bending_compliance=float(
                material_config.get("bending_compliance", 1e-4)
            ),
            stretch_relaxation=float(
                material_config.get("stretch_relaxation", 0.3)
            ),
            bending_relaxation=float(
                material_config.get("bending_relaxation", 0.1)
            ),
            air_resistance=float(material_config.get("air_resistance", 1e-3)),
        ),
    )
    camera = scene.add_camera(
        res=camera_res,
        pos=camera_pos,
        lookat=camera_lookat,
        fov=DEFAULT_CAMERA_FOV,
    )
    scene.build()

    config = {
        "asset_path": str(asset_dir),
        "manifest": str(manifest_path),
        "manifest_scale": float(manifest.get("scale", 1.0)),
        "init_height": init_height,
        "duration_seconds": duration_seconds,
        "sim_steps": sim_steps,
        "shake": shake,
        "shake_steps": shake_steps,
        "shake_amplitude": shake_amplitude,
        "device": device,
        "resolved_backend": backend_name,
        "backend_warning": backend_warning,
        "headless": headless,
        "fps": fps,
        "render_interval": render_interval,
        "camera_res": list(camera_res),
        "camera_pos": list(camera_pos),
        "camera_lookat": list(camera_lookat),
        "camera_fov": DEFAULT_CAMERA_FOV,
        "gravity": list(gravity),
        "background_color": list(background_color),
        "ambient_light": list(ambient_light),
        "floor_size": list(floor_size),
        "floor_tile_size": list(floor_tile_size),
        "floor_color": list(floor_color),
        "backdrop": backdrop,
        "backdrop_color": list(backdrop_color),
    }
    _write_json(config_path, config)

    rest_positions = cloth.get_particles_pos().clone()
    particle_frames = []
    rendered_frames = 0
    camera.start_recording()
    for step in tqdm(range(sim_steps), desc="Genesis Cloth Simulation"):
        if shake and step < shake_steps:
            _shake_cloth_particles(
                gs,
                cloth,
                rest_positions,
                step,
                shake_steps,
                shake_amplitude,
            )
        scene.step()
        if step % render_interval == 0:
            particle_frames.append(_tensor_to_numpy(cloth.get_particles_pos()))
            camera.render()
            rendered_frames += 1
    _save_camera_recording(camera, video_path, fps)
    _save_particle_positions(
        particle_positions_path=particle_positions_path,
        particle_frames=particle_frames,
        rest_positions=rest_positions,
        fps=fps,
        render_interval=render_interval,
    )

    summary = {
        "video": str(video_path),
        "particle_positions": str(particle_positions_path),
        "rendered_frames": rendered_frames,
        "particles": int(cloth.n_particles),
        "sim_steps": sim_steps,
        "duration_seconds": duration_seconds,
        "requested_device": device,
        "resolved_backend": backend_name,
        "backend_warning": backend_warning,
        "status": "completed",
    }
    _write_json(summary_path, summary)
    logger.info(
        "Genesis cloth simulation outputs: "
        f"video {video_path}, particles {particle_positions_path}, "
        f"config {config_path}, summary {summary_path}"
    )
    return str(video_path)


def entrypoint(**kwargs) -> str:
    if kwargs:
        cfg = GenesisClothSimConfig(**kwargs)
    else:
        cfg = tyro.cli(GenesisClothSimConfig)

    return simulate_genesis_cloth(
        asset_path=cfg.asset_path,
        output_dir=cfg.output_dir,
        init_height=cfg.init_height,
        duration_seconds=cfg.duration_seconds,
        sim_steps=cfg.sim_steps,
        shake=cfg.shake,
        shake_steps=cfg.shake_steps,
        shake_amplitude=cfg.shake_amplitude,
        device=cfg.device,
        headless=cfg.headless,
        fps=cfg.fps,
        render_interval=cfg.render_interval,
        camera_res=cfg.camera_res,
        gravity=cfg.gravity,
        background_color=cfg.background_color,
        ambient_light=cfg.ambient_light,
        floor_size=cfg.floor_size,
        floor_tile_size=cfg.floor_tile_size,
        floor_color=cfg.floor_color,
        backdrop=cfg.backdrop,
        backdrop_color=cfg.backdrop_color,
    )


def _shake_cloth_particles(
    gs: Any,
    cloth: Any,
    rest_positions: Any,
    step: int,
    shake_steps: int,
    amplitude: float,
) -> None:
    if amplitude <= 0.0 or shake_steps <= 0:
        return

    positions = cloth.get_particles_pos().clone()
    min_values = rest_positions.min(dim=0).values
    max_values = rest_positions.max(dim=0).values
    height = gs.torch.clamp(max_values[2] - min_values[2], min=1e-6)
    y_values = rest_positions[:, 1]
    z_values = rest_positions[:, 2]

    top_ratio = gs.torch.clamp((max_values[2] - z_values) / height, 0.0, 1.0)
    anchor_mask = top_ratio < 0.01
    wind_strength = (
        gs.torch.clamp((top_ratio - 0.005) / 0.995, 0.0, 1.0) ** 1.1
    )

    progress = float(step) / max(float(shake_steps), 1.0)
    phase = 2.0 * math.pi * 2.2 * progress
    traveling_phase = phase - 0.85 * top_ratio
    center_y = rest_positions[:, 1].mean()
    width = gs.torch.clamp(max_values[1] - min_values[1], min=1e-6)
    across = (y_values - center_y) / width

    offsets = gs.torch.zeros_like(rest_positions)
    offsets[:, 0] = (
        float(amplitude) * 2.25 * gs.torch.sin(traveling_phase) * wind_strength
    )
    offsets[:, 1] = (
        float(amplitude)
        * 0.04
        * gs.torch.sin(traveling_phase * 1.10 + across)
        * wind_strength
    )
    offsets[:, 2] = (
        float(amplitude)
        * 0.12
        * gs.torch.sin(traveling_phase * 1.15 + across)
        * wind_strength
    )

    target_positions = rest_positions + offsets
    blend = (0.58 * wind_strength).unsqueeze(1)
    positions = positions + (target_positions - positions) * blend
    positions[anchor_mask] = rest_positions[anchor_mask]
    cloth.set_particles_pos(positions)


def _add_scene_floor_and_backdrop(
    gs: Any,
    scene: Any,
    camera_pos: tuple[float, float, float],
    camera_lookat: tuple[float, float, float],
    floor_size: tuple[float, float],
    floor_tile_size: tuple[float, float],
    floor_color: tuple[float, float, float, float],
    backdrop: bool,
    backdrop_color: tuple[float, float, float, float],
) -> None:
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, 0.0),
            plane_size=tuple(float(value) for value in floor_size),
            tile_size=tuple(float(value) for value in floor_tile_size),
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Rough(
            color=tuple(float(value) for value in floor_color)
        ),
        name="floor",
    )

    if not backdrop:
        return

    normal = _horizontal_unit_vector(
        (
            camera_pos[0] - camera_lookat[0],
            camera_pos[1] - camera_lookat[1],
            0.0,
        )
    )
    camera_distance = math.sqrt(
        (camera_pos[0] - camera_lookat[0]) ** 2
        + (camera_pos[1] - camera_lookat[1]) ** 2
        + (camera_pos[2] - camera_lookat[2]) ** 2
    )
    backdrop_distance = max(1.5, camera_distance * 0.72)
    backdrop_pos = (
        camera_lookat[0] - normal[0] * backdrop_distance,
        camera_lookat[1] - normal[1] * backdrop_distance,
        camera_lookat[2],
    )
    backdrop_size = (
        max(float(floor_size[0]), float(floor_size[1])),
        max(float(floor_size[0]), float(floor_size[1])) * 0.7,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=backdrop_pos,
            normal=normal,
            plane_size=backdrop_size,
            collision=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Rough(
            color=tuple(float(value) for value in backdrop_color)
        ),
        name="backdrop",
    )


def _horizontal_unit_vector(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if length <= 1e-6:
        return (1.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, 0.0)


def _save_camera_recording(camera: Any, video_path: Path, fps: int) -> None:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    if not camera._in_recording:
        raise RuntimeError("Camera recording was not started.")
    if not camera._recorded_imgs:
        raise RuntimeError("No frames were recorded.")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(list(camera._recorded_imgs), fps=fps)
    try:
        clip.write_videofile(
            str(video_path),
            fps=fps,
            logger=None,
            codec="libx264",
            preset="ultrafast",
        )
    finally:
        close = getattr(clip, "close", None)
        if close is not None:
            close()
        camera._recorded_t_prev = -1
        camera._recorded_imgs.clear()
        camera._in_recording = False


def _save_particle_positions(
    particle_positions_path: Path,
    particle_frames: list[Any],
    rest_positions: Any,
    fps: int,
    render_interval: int,
) -> None:
    import numpy as np

    if not particle_frames:
        raise RuntimeError("No particle frames were recorded.")

    particle_positions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        particle_positions_path,
        positions=np.stack(particle_frames, axis=0).astype(np.float32),
        rest_positions=_tensor_to_numpy(rest_positions).astype(np.float32),
        fps=np.array(fps, dtype=np.int32),
        render_interval=np.array(render_interval, dtype=np.int32),
    )


def _tensor_to_numpy(tensor: Any) -> Any:
    array = tensor.detach() if hasattr(tensor, "detach") else tensor
    array = array.cpu() if hasattr(array, "cpu") else array
    array = array.numpy() if hasattr(array, "numpy") else array
    return array


def _collar_handle_weights(gs: Any, rest_positions: Any) -> Any:
    min_values = rest_positions.min(dim=0).values
    max_values = rest_positions.max(dim=0).values
    center = rest_positions.mean(dim=0)
    width = gs.torch.clamp(max_values[1] - min_values[1], min=1e-6)
    height = gs.torch.clamp(max_values[2] - min_values[2], min=1e-6)

    y_values = rest_positions[:, 1]
    z_values = rest_positions[:, 2]
    top_start = max_values[2] - 0.32 * height
    top_weights = gs.torch.clamp(
        (z_values - top_start) / (0.32 * height), 0.0, 1.0
    )

    left_center = center[1] - 0.22 * width
    right_center = center[1] + 0.22 * width
    sigma = 0.12 * width
    left_weights = gs.torch.exp(-0.5 * ((y_values - left_center) / sigma) ** 2)
    right_weights = gs.torch.exp(
        -0.5 * ((y_values - right_center) / sigma) ** 2
    )
    side_weights = gs.torch.clamp(left_weights + right_weights, 0.0, 1.0)
    return top_weights * side_weights


def _front_camera_pose(
    manifest: dict[str, Any], init_height: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    stats = manifest.get("mesh_stats", {})
    bbox_min = stats.get("bbox_min", [-0.5, -0.5, -0.5])
    bbox_max = stats.get("bbox_max", [0.5, 0.5, 0.5])
    extents = stats.get("extents", [1.0, 1.0, 1.0])

    center = [
        0.5 * (float(bbox_min[index]) + float(bbox_max[index]))
        for index in range(3)
    ]
    width = float(extents[1])
    height = float(extents[2])
    depth = max(float(extents[0]), 0.05)
    distance = max(width, height) * 2.45 + depth

    lookat = (center[0], center[1], center[2] + float(init_height))
    pos = (
        center[0] + distance,
        center[1] - 0.22 * distance,
        lookat[2] + 0.04 * height,
    )
    return pos, lookat


def _init_genesis_backend(gs: Any, device: str) -> tuple[str, str | None]:
    normalized = device.lower()
    if normalized == "cpu":
        gs.init(backend=gs.cpu, logging_level="warning")
        return "cpu", None
    if normalized == "cuda":
        gs.init(backend=gs.cuda, logging_level="warning")
        return "cuda", None
    if normalized != "auto":
        raise ValueError(
            f"Unsupported Genesis backend: {device}. Use 'cpu', 'cuda', or 'auto'."
        )

    try:
        gs.init(backend=gs.cuda, logging_level="warning")
        return "cuda", None
    except Exception as exc:
        warning = (
            "Genesis CUDA backend is unavailable; falling back to CPU. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
        gs.init(backend=gs.cpu, logging_level="warning")
        return "cpu", warning


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    entrypoint()
