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
from dataclasses import asdict, dataclass, field
from typing import Optional

import imageio
import numpy as np
import sapien.core as sapien
import torch
import trimesh
import tyro
from embodied_gen.utils.log import logger
from embodied_gen.utils.simulation import (
    FrankaPandaGrasper,
    SapienSceneManager,
    capture_frame,
    create_panda_agent,
    create_recording_camera,
    estimate_grasp_width,
    get_actor_bottom_z,
    get_actor_mesh,
    load_actor_from_urdf,
    load_collision_mesh_from_urdf,
    quat_from_yaw,
    set_ground_base_color,
)

GROUND_BASE_COLOR = [0.78, 0.90, 0.72, 1.0]
SETTLE_CHECK_INTERVAL = 10
SETTLE_STABLE_WINDOWS = 3
SETTLE_BOTTOM_Z_TOL = 5e-4
MAX_EXTRA_SETTLE_STEPS = 120


@dataclass
class TrialResult:
    """Stores per-trial evaluation results."""

    yaw_deg: float
    success: bool
    scale_factor: float
    object_width_before_scale: float
    object_width_after_scale: float
    settled_bottom_z: float
    final_bottom_z: float
    lift_delta: float
    peak_bottom_z: float | None = None
    peak_lift_delta: float | None = None
    lift_success_threshold: float | None = None
    tcp_object_offset_range: float | None = None
    sync_tol: float | None = None
    final_lift_pass: bool | None = None
    sync_pass: bool | None = None
    video_path: str | None = None
    note: str = ""


@dataclass
class EvalCollisionConfig:
    urdf_path: str
    output_path: Optional[str] = None
    num_trials: int = 4
    max_gripper_width: float = 0.075
    gripper_clearance_ratio: float = 0.9
    sim_freq: int = 200
    control_freq: int = 20
    settle_steps: int = 240
    post_grasp_steps: int = 80
    lift_success_height: Optional[float] = None
    lift_success_ratio: float = 0.01
    min_lift_success_height: float = 0.001
    max_lift_success_height: float = 0.003
    sync_tol: float = 0.02
    approach_offset: float = 0.06
    grasp_clearance: float = 0.004
    grasp_height_ratio: float = 0.45
    max_descent_from_top: float = 0.03
    hover_offset: float = 0.12
    hover_open_steps: int = 10
    descent_stage_count: int = 4
    descent_n_max_step: int = 25
    lift_distance: float = 0.10
    close_steps: int = 20
    object_x: float = 0.55
    object_y: float = 0.0
    z_offset: float = 0.005
    sim_backend: str = "cpu"
    render_backend: str = "gpu"
    ray_tracing: bool = False
    save_video: bool = True
    video_path: Optional[str] = None
    video_fps: int = 20
    render_interval: int = 4
    image_hw: tuple[int, int] = (512, 512)

    def __post_init__(self) -> None:
        output_dir = os.path.join(
            os.path.dirname(self.urdf_path), "grasp_trial"
        )
        if self.output_path is None:
            self.output_path = os.path.join(
                output_dir, "collision_success_eval.json"
            )
        if self.video_path is None:
            self.video_path = os.path.join(
                output_dir, "collision_success_eval.mp4"
            )


def _compute_scale_factor(
    urdf_path: str,
    max_gripper_width: float,
    clearance_ratio: float,
) -> tuple[float, float]:
    """Compute a scale that fits the asset within the gripper width."""
    mesh = load_collision_mesh_from_urdf(urdf_path)
    grasp_width = estimate_grasp_width(mesh)
    target_width = max_gripper_width * clearance_ratio
    if grasp_width <= 1e-6:
        raise ValueError(f"Invalid grasp width estimated from {urdf_path}")

    scale = min(1.0, target_width / grasp_width)
    return float(scale), float(grasp_width)


def _compute_spawn_center_z(
    mesh: trimesh.Trimesh,
    scale_factor: float,
    z_offset: float,
) -> float:
    """Compute actor-center z so the scaled mesh bottom is z_offset above z=0."""
    local_bottom_z = float(mesh.bounds[0, 2] * scale_factor)
    return z_offset - local_bottom_z


def _compute_adaptive_lift_threshold(
    actor: sapien.Entity,
    ratio: float,
    min_height: float,
    max_height: float,
    absolute_override: float | None = None,
) -> float:
    """Compute a robust lift threshold from the settled object height."""
    if absolute_override is not None:
        return float(absolute_override)

    mesh = get_actor_mesh(actor)
    object_height = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])
    adaptive_height = object_height * ratio
    return float(np.clip(adaptive_height, min_height, max_height))


def _build_trial_video_path(
    video_path: str,
    trial_idx: int,
    yaw_deg: float,
) -> str:
    """Build a unique per-trial video path from the base output path."""
    root, ext = os.path.splitext(video_path)
    if not ext:
        ext = ".mp4"
    return f"{root}_trial{trial_idx:02d}_yaw{int(round(yaw_deg)):03d}{ext}"


@dataclass
class _GraspTracker:
    """Tracks gripper-object sync metrics during the grasp/lift phase.

    The lift_delta of the object alone is fragile: a bounced-away object can
    momentarily rise high before falling back. By logging the per-step offset
    between the object bottom and the gripper TCP, we can also verify that the
    object actually moves together with the gripper after closing.
    """

    actor: sapien.Entity
    grasper: FrankaPandaGrasper
    peak_bottom_z: float | None = None
    tcp_object_offsets: list[float] = field(default_factory=list)

    def update(self) -> None:
        bottom_z = get_actor_bottom_z(self.actor)
        tcp_z = float(self.grasper.agent.tcp.pose[0].sp.p[2])
        self.peak_bottom_z = (
            bottom_z
            if self.peak_bottom_z is None
            else max(self.peak_bottom_z, bottom_z)
        )
        self.tcp_object_offsets.append(bottom_z - tcp_z)

    @property
    def offset_range(self) -> float:
        if not self.tcp_object_offsets:
            return 0.0
        return float(
            max(self.tcp_object_offsets) - min(self.tcp_object_offsets)
        )


def _execute_actions(
    scene_manager: SapienSceneManager,
    agent: object,
    actions: np.ndarray,
    control_freq: int,
    camera: sapien.render.RenderCameraComponent | None = None,
    render_interval: int = 1,
    video_frames: list[np.ndarray] | None = None,
    tracker: _GraspTracker | None = None,
) -> None:
    """Run a sequence of robot actions."""
    sim_steps = max(1, scene_manager.sim_freq // control_freq)
    cameras = [] if camera is None else [camera]
    render_keys = [] if camera is None else ["Color"]
    for idx, action in enumerate(actions):
        frames = scene_manager.step_action(
            agent,
            torch.tensor(action[None, ...], dtype=torch.float32),
            cameras=cameras,
            render_keys=render_keys,
            sim_steps_per_control=sim_steps,
        )
        if (
            camera is not None
            and video_frames is not None
            and idx % max(1, render_interval) == 0
        ):
            video_frames.append(np.array(frames[camera.name][0]["Color"]))
        if tracker is not None:
            tracker.update()


def _hold_gripper_state(
    scene_manager: SapienSceneManager,
    grasper: FrankaPandaGrasper,
    gripper_state: int,
    control_freq: int,
    n_step: int,
    camera: sapien.render.RenderCameraComponent | None = None,
    render_interval: int = 1,
    video_frames: list[np.ndarray] | None = None,
    tracker: _GraspTracker | None = None,
) -> None:
    """Hold gripper open/close while stepping the scene."""
    hold_actions = grasper.control_gripper(
        gripper_state=gripper_state,
        n_step=n_step,
    )
    _execute_actions(
        scene_manager,
        grasper.agent,
        hold_actions,
        control_freq,
        camera=camera,
        render_interval=render_interval,
        video_frames=video_frames,
        tracker=tracker,
    )


def _wait_until_actor_settled(
    scene_manager: SapienSceneManager,
    grasper: FrankaPandaGrasper,
    actor: sapien.Entity,
    control_freq: int,
    initial_bottom_z: float,
    max_extra_steps: int = MAX_EXTRA_SETTLE_STEPS,
    check_interval: int = SETTLE_CHECK_INTERVAL,
    stable_windows: int = SETTLE_STABLE_WINDOWS,
    bottom_z_tol: float = SETTLE_BOTTOM_Z_TOL,
    camera: sapien.render.RenderCameraComponent | None = None,
    render_interval: int = 1,
    video_frames: list[np.ndarray] | None = None,
) -> float:
    """Wait until the dropped object is visually settled on the ground."""
    remaining_steps = max(0, max_extra_steps)
    previous_bottom_z = initial_bottom_z
    stable_count = 0

    while remaining_steps > 0 and stable_count < stable_windows:
        n_step = min(check_interval, remaining_steps)
        _hold_gripper_state(
            scene_manager,
            grasper,
            gripper_state=1,
            control_freq=control_freq,
            n_step=n_step,
            camera=camera,
            render_interval=render_interval,
            video_frames=video_frames,
        )
        current_bottom_z = get_actor_bottom_z(actor)
        if abs(current_bottom_z - previous_bottom_z) <= bottom_z_tol:
            stable_count += 1
        else:
            stable_count = 0
        previous_bottom_z = current_bottom_z
        remaining_steps -= n_step

    return previous_bottom_z


def _plan_scripted_grasp_stages(
    grasper: FrankaPandaGrasper,
    actor: sapien.Entity,
    grasp_height_ratio: float,
    grasp_clearance: float,
    approach_offset: float,
    lift_distance: float,
    max_descent_from_top: float | None = None,
) -> tuple[sapien.Pose, sapien.Pose, sapien.Pose]:
    """Plan a simple top-down scripted grasp."""
    mesh = get_actor_mesh(actor)
    bounds = mesh.bounds
    approaching = np.array([0.0, 0.0, -1.0])
    center = bounds.mean(axis=0)
    extents_xy = bounds[1, :2] - bounds[0, :2]
    closing = (
        np.array([1.0, 0.0, 0.0])
        if extents_xy[0] <= extents_xy[1]
        else np.array([0.0, 1.0, 0.0])
    )
    object_height = bounds[1, 2] - bounds[0, 2]
    grasp_z = bounds[0, 2] + object_height * grasp_height_ratio
    if max_descent_from_top is not None:
        grasp_z = max(grasp_z, bounds[1, 2] - max_descent_from_top)
    grasp_z = float(
        np.clip(
            grasp_z,
            bounds[0, 2] + 0.015,
            bounds[1, 2] - 0.005,
        )
    )
    center = np.array([center[0], center[1], grasp_z + grasp_clearance])
    grasp_pose = grasper.agent.build_grasp_pose(approaching, closing, center)
    pre_grasp_pose = sapien.Pose(
        p=grasp_pose.p + np.array([0.0, 0.0, approach_offset]),
        q=grasp_pose.q,
    )
    lift_pose = sapien.Pose(
        p=grasp_pose.p + np.array([0.0, 0.0, lift_distance]),
        q=grasp_pose.q,
    )

    return pre_grasp_pose, grasp_pose, lift_pose


def _build_grasp_stage_candidates(
    grasper: FrankaPandaGrasper,
    actor: sapien.Entity,
    grasp_height_ratio: float,
    grasp_clearance: float,
    approach_offset: float,
    lift_distance: float,
    max_descent_from_top: float | None = None,
) -> list[tuple[float, float, sapien.Pose, sapien.Pose, sapien.Pose]]:
    """Build fallback grasp-stage candidates for tapered objects like bottles."""
    ratio_candidates = [
        grasp_height_ratio,
        min(0.95, grasp_height_ratio + 0.08),
        min(0.95, grasp_height_ratio + 0.16),
    ]
    clearance_candidates = [
        grasp_clearance,
        grasp_clearance + 0.004,
        grasp_clearance + 0.008,
    ]
    candidates = []
    seen_keys = set()
    for ratio, clearance in zip(ratio_candidates, clearance_candidates):
        key = (round(ratio, 4), round(clearance, 4))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pre_grasp_pose, grasp_pose, lift_pose = _plan_scripted_grasp_stages(
            grasper,
            actor,
            grasp_height_ratio=ratio,
            grasp_clearance=clearance,
            approach_offset=approach_offset,
            lift_distance=lift_distance,
            max_descent_from_top=max_descent_from_top,
        )
        candidates.append(
            (ratio, clearance, pre_grasp_pose, grasp_pose, lift_pose)
        )

    return candidates


def _build_hover_pose(
    actor: sapien.Entity,
    grasp_pose: sapien.Pose,
    hover_offset: float,
) -> sapien.Pose:
    """Build a hover pose at a fixed offset above the object top surface."""
    mesh = get_actor_mesh(actor)
    top_z = float(mesh.bounds[1, 2])
    return sapien.Pose(
        p=np.array([grasp_pose.p[0], grasp_pose.p[1], top_z + hover_offset]),
        q=grasp_pose.q,
    )


def _build_descent_stage_poses(
    grasp_pose: sapien.Pose,
    hover_offset: float,
    num_stages: int,
) -> list[sapien.Pose]:
    """Split the downward approach into multiple slow open-gripper stages."""
    if num_stages <= 0:
        return [grasp_pose]

    stage_offsets = np.linspace(hover_offset, 0.0, num_stages + 1)[1:]
    return [
        sapien.Pose(
            p=grasp_pose.p + np.array([0.0, 0.0, float(offset)]),
            q=grasp_pose.q,
        )
        for offset in stage_offsets
    ]


def run_single_trial(
    args: EvalCollisionConfig,
    yaw_deg: float,
    scale_factor: float,
    grasp_width: float,
    record_video: bool = False,
    video_path: str | None = None,
) -> TrialResult:
    """Run one grasp trial with a fixed yaw."""
    scene_manager = SapienSceneManager(
        sim_freq=args.sim_freq,
        ray_tracing=args.ray_tracing,
        device=args.sim_backend,
    )
    scene = scene_manager.scene
    set_ground_base_color(scene, GROUND_BASE_COLOR)
    agent = create_panda_agent(
        scene,
        control_freq=args.control_freq,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    video_frames: list[np.ndarray] | None = None
    camera = None
    if record_video:
        video_frames = []
        camera = create_recording_camera(
            scene_manager,
            eye_pos=[args.object_x - 0.32, args.object_y - 0.52, 0.56],
            target_pt=[args.object_x - 0.01, args.object_y, 0.27],
            image_hw=tuple(args.image_hw),
            fovy_deg=60.0,
        )

    collision_mesh = load_collision_mesh_from_urdf(args.urdf_path)
    spawn_center_z = _compute_spawn_center_z(
        collision_mesh,
        scale_factor=scale_factor,
        z_offset=args.z_offset,
    )
    spawn_pose = sapien.Pose(
        p=[
            args.object_x,
            args.object_y,
            spawn_center_z,
        ],
        q=quat_from_yaw(yaw_deg),
    )
    actor = load_actor_from_urdf(
        scene,
        args.urdf_path,
        pose=spawn_pose,
        use_static=False,
        update_mass=True,
        scale=scale_factor,
    )

    if video_frames is not None and camera is not None:
        video_frames.append(capture_frame(scene, camera))
    grasper = FrankaPandaGrasper(agent, control_freq=args.control_freq)
    _hold_gripper_state(
        scene_manager,
        grasper,
        gripper_state=1,
        control_freq=args.control_freq,
        n_step=max(
            1,
            args.settle_steps
            // max(1, scene_manager.sim_freq // args.control_freq),
        ),
        camera=camera,
        render_interval=args.render_interval,
        video_frames=video_frames,
    )
    settled_bottom_z = get_actor_bottom_z(actor)
    settled_bottom_z = _wait_until_actor_settled(
        scene_manager,
        grasper,
        actor,
        control_freq=args.control_freq,
        initial_bottom_z=settled_bottom_z,
        camera=camera,
        render_interval=args.render_interval,
        video_frames=video_frames,
    )
    grasp_candidates = _build_grasp_stage_candidates(
        grasper,
        actor,
        grasp_height_ratio=args.grasp_height_ratio,
        grasp_clearance=args.grasp_clearance,
        approach_offset=args.approach_offset,
        lift_distance=args.lift_distance,
        max_descent_from_top=args.max_descent_from_top,
    )
    selected_lift_pose = None
    selected_candidate_note = ""
    grasp_stage_failure_note = "failed to reach pre-grasp pose"
    for candidate_idx, candidate in enumerate(grasp_candidates):
        (
            candidate_ratio,
            candidate_clearance,
            _pre_grasp_pose,
            grasp_pose,
            lift_pose,
        ) = candidate
        hover_pose = _build_hover_pose(
            actor, grasp_pose, hover_offset=args.hover_offset
        )
        hover_actions = grasper.move_to_pose(
            hover_pose,
            grasper.control_timestep,
            gripper_state=1,
            n_max_step=80,
        )
        if hover_actions is None:
            grasp_stage_failure_note = "failed to reach hover pose"
            continue
        _execute_actions(
            scene_manager,
            agent,
            hover_actions,
            args.control_freq,
            camera=camera,
            render_interval=args.render_interval,
            video_frames=video_frames,
        )
        _hold_gripper_state(
            scene_manager,
            grasper,
            gripper_state=1,
            control_freq=args.control_freq,
            n_step=args.hover_open_steps,
            camera=camera,
            render_interval=args.render_interval,
            video_frames=video_frames,
        )

        descent_failed = False
        for descent_pose in _build_descent_stage_poses(
            grasp_pose,
            hover_offset=args.hover_offset,
            num_stages=args.descent_stage_count,
        ):
            descent_actions = grasper.move_to_pose(
                descent_pose,
                grasper.control_timestep,
                gripper_state=1,
                n_max_step=args.descent_n_max_step,
            )
            if descent_actions is None:
                descent_failed = True
                grasp_stage_failure_note = (
                    "failed during slow descent to grasp pose"
                )
                break
            _execute_actions(
                scene_manager,
                agent,
                descent_actions,
                args.control_freq,
                camera=camera,
                render_interval=args.render_interval,
                video_frames=video_frames,
            )
        if descent_failed:
            continue

        _hold_gripper_state(
            scene_manager,
            grasper,
            gripper_state=1,
            control_freq=args.control_freq,
            n_step=2,
            camera=camera,
            render_interval=args.render_interval,
            video_frames=video_frames,
        )
        selected_lift_pose = lift_pose
        selected_candidate_note = (
            ""
            if candidate_idx == 0
            else (
                f"fallback grasp candidate ratio={candidate_ratio:.2f}, "
                f"clearance={candidate_clearance:.3f}"
            )
        )
        break

    if selected_lift_pose is None:
        if video_frames is not None and video_path is not None:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, video_frames, fps=args.video_fps)
        return TrialResult(
            yaw_deg=yaw_deg,
            success=False,
            scale_factor=scale_factor,
            object_width_before_scale=grasp_width,
            object_width_after_scale=grasp_width * scale_factor,
            settled_bottom_z=settled_bottom_z,
            final_bottom_z=settled_bottom_z,
            lift_delta=0.0,
            video_path=video_path,
            note=grasp_stage_failure_note,
        )

    lift_success_threshold = _compute_adaptive_lift_threshold(
        actor,
        ratio=args.lift_success_ratio,
        min_height=args.min_lift_success_height,
        max_height=args.max_lift_success_height,
        absolute_override=args.lift_success_height,
    )
    tracker = _GraspTracker(actor=actor, grasper=grasper)
    close_actions = grasper.control_gripper(
        gripper_state=-1,
        n_step=args.close_steps,
    )
    _execute_actions(
        scene_manager,
        agent,
        close_actions,
        args.control_freq,
        camera=camera,
        render_interval=args.render_interval,
        video_frames=video_frames,
        tracker=tracker,
    )

    stage_note = "ok"
    lift_actions = grasper.move_to_pose(
        selected_lift_pose,
        grasper.control_timestep,
        gripper_state=-1,
        n_max_step=50,
    )
    if lift_actions is not None:
        _execute_actions(
            scene_manager,
            agent,
            lift_actions,
            args.control_freq,
            camera=camera,
            render_interval=args.render_interval,
            video_frames=video_frames,
            tracker=tracker,
        )
    else:
        stage_note = "failed to lift after closing"
    _hold_gripper_state(
        scene_manager,
        grasper,
        gripper_state=-1,
        control_freq=args.control_freq,
        n_step=args.post_grasp_steps,
        camera=camera,
        render_interval=args.render_interval,
        video_frames=video_frames,
        tracker=tracker,
    )

    final_bottom_z = get_actor_bottom_z(actor)
    lift_delta = final_bottom_z - settled_bottom_z
    peak_bottom_z = (
        final_bottom_z
        if tracker.peak_bottom_z is None
        else tracker.peak_bottom_z
    )
    peak_lift_delta = peak_bottom_z - settled_bottom_z
    offset_range = tracker.offset_range
    final_lift_pass = bool(lift_delta >= lift_success_threshold)
    sync_pass = bool(offset_range <= args.sync_tol)
    success = bool(final_lift_pass and sync_pass)
    if video_frames is not None and camera is not None:
        video_frames.append(capture_frame(scene, camera))
    if video_frames is not None and video_path is not None:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, video_frames, fps=args.video_fps)

    if stage_note != "ok":
        note = stage_note
    elif success:
        note = selected_candidate_note or "ok"
    elif not final_lift_pass and not sync_pass:
        note = "object dropped and decoupled from gripper"
    elif not final_lift_pass:
        note = "object did not stay lifted (likely bounced or dropped)"
    else:
        note = "object did not move synchronously with gripper"

    return TrialResult(
        yaw_deg=yaw_deg,
        success=success,
        scale_factor=scale_factor,
        object_width_before_scale=grasp_width,
        object_width_after_scale=grasp_width * scale_factor,
        settled_bottom_z=settled_bottom_z,
        final_bottom_z=final_bottom_z,
        lift_delta=lift_delta,
        peak_bottom_z=peak_bottom_z,
        peak_lift_delta=peak_lift_delta,
        lift_success_threshold=lift_success_threshold,
        tcp_object_offset_range=offset_range,
        sync_tol=args.sync_tol,
        final_lift_pass=final_lift_pass,
        sync_pass=sync_pass,
        video_path=video_path,
        note=note,
    )


def entrypoint(**kwargs) -> dict:
    """Run collision-success evaluation for a URDF asset."""
    if kwargs:
        kwargs.setdefault("urdf_path", "__dummy__.urdf")
        args = EvalCollisionConfig(**kwargs)
    else:
        args = tyro.cli(EvalCollisionConfig)

    if not os.path.exists(args.urdf_path):
        raise FileNotFoundError(f"URDF file not found: {args.urdf_path}")

    logger.info(
        f"Start collision-success eval: urdf={args.urdf_path}, "
        f"num_trials={args.num_trials}, sync_tol={args.sync_tol}, "
        f"output={args.output_path}"
    )
    scale_factor, grasp_width = _compute_scale_factor(
        args.urdf_path,
        max_gripper_width=args.max_gripper_width,
        clearance_ratio=args.gripper_clearance_ratio,
    )
    yaw_values = np.linspace(0, 360, args.num_trials, endpoint=False)
    trials = [
        run_single_trial(
            args,
            float(yaw_deg),
            scale_factor,
            grasp_width,
            record_video=args.save_video,
            video_path=(
                _build_trial_video_path(args.video_path, idx, float(yaw_deg))
                if args.save_video
                else None
            ),
        )
        for idx, yaw_deg in enumerate(yaw_values)
    ]

    success_count = sum(int(trial.success) for trial in trials)
    result = {
        "urdf_path": args.urdf_path,
        "num_trials": args.num_trials,
        "num_success": success_count,
        "collision_success_rate": success_count / max(1, args.num_trials),
        "scale_factor": scale_factor,
        "estimated_grasp_width_before_scale": grasp_width,
        "estimated_grasp_width_after_scale": grasp_width * scale_factor,
        "video_path": args.video_path if args.save_video else None,
        "trial_video_paths": [
            trial.video_path
            for trial in trials
            if trial.video_path is not None
        ],
        "trials": [asdict(trial) for trial in trials],
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Collision success report saved to {args.output_path}")

    return result


if __name__ == "__main__":
    entrypoint()
