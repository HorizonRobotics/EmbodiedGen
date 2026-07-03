import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--particle-positions", required=True)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--tmp-video", required=True)
    args = parser.parse_args()

    render_mesh_displacement_heatmap(
        video_path=Path(args.video_path),
        mesh_path=Path(args.mesh_path),
        particle_positions_path=Path(args.particle_positions),
        sim_config_path=Path(args.sim_config),
        output_image_path=Path(args.output_image),
        output_video_path=Path(args.output_video),
        tmp_video_path=Path(args.tmp_video),
    )


def render_mesh_displacement_heatmap(
    video_path: Path,
    mesh_path: Path,
    particle_positions_path: Path,
    sim_config_path: Path,
    output_image_path: Path,
    output_video_path: Path,
    tmp_video_path: Path,
) -> None:
    data = np.load(particle_positions_path)
    positions = data["positions"].astype(np.float32)
    rest_positions = data["rest_positions"].astype(np.float32)
    config = json.loads(sim_config_path.read_text())

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"Invalid particle positions shape: {positions.shape}"
        )
    if rest_positions.shape != positions.shape[1:]:
        raise ValueError(
            "rest_positions must have shape "
            f"{positions.shape[1:]}, got {rest_positions.shape}"
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or float(config.get("fps", 30))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(tmp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not write video: {tmp_video_path}")

    displacement = np.linalg.norm(
        positions - rest_positions[None, :, :], axis=2
    )
    scale = (
        float(np.percentile(displacement, 95)) if displacement.size else 1.0
    )
    scale = max(scale, 1e-6)

    camera_pos = np.asarray(config["camera_pos"], dtype=np.float32)
    camera_lookat = np.asarray(config["camera_lookat"], dtype=np.float32)
    camera_fov = float(config["camera_fov"])
    inset_bbox = _compute_global_inset_bbox(
        positions=positions,
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
        camera_fov=camera_fov,
        image_size=(width, height),
    )

    snapshot_index = min(
        positions.shape[0] - 1,
        max(0, int(round(fps)) - 1),
    )

    frame_index = 0
    snapshot_frame = None
    try:
        while frame_index < positions.shape[0]:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            heatmap = _compose_displacement_frame(
                frame_rgb=frame_rgb,
                positions=positions[frame_index],
                displacement=displacement[frame_index],
                scale=scale,
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=camera_fov,
                inset_bbox=inset_bbox,
            )
            if frame_index == snapshot_index:
                snapshot_frame = heatmap
            writer.write(cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0 or snapshot_frame is None:
        raise RuntimeError("No frames were written to mesh displacement video")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_image_path), cv2.cvtColor(snapshot_frame, cv2.COLOR_RGB2BGR)
    )


def _compose_displacement_frame(
    frame_rgb: np.ndarray,
    positions: np.ndarray,
    displacement: np.ndarray,
    scale: float,
    camera_pos: np.ndarray,
    camera_lookat: np.ndarray,
    camera_fov: float,
    inset_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    pixels, depths = _project_points(
        points=positions,
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
        camera_fov=camera_fov,
        image_size=(width, height),
    )
    normalized = np.clip(displacement / scale, 0.0, 1.0)
    scalar_layer, weight_layer = _rasterize_particle_scalar_field(
        pixels=pixels,
        depths=depths,
        normalized=normalized,
        image_size=(width, height),
    )
    scalar_blur, weight_blur = _smooth_scalar_field(scalar_layer, weight_layer)
    return _compose_rgb_with_inset(
        frame_rgb=frame_rgb,
        scalar_field=scalar_blur,
        weight_field=weight_blur,
        scale=scale,
        inset_bbox=inset_bbox,
    )


def _compute_global_inset_bbox(
    positions: np.ndarray,
    camera_pos: np.ndarray,
    camera_lookat: np.ndarray,
    camera_fov: float,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    pad = 28
    flat = positions.reshape(-1, 3)
    pixels, depths = _project_points(
        points=flat,
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
        camera_fov=camera_fov,
        image_size=image_size,
    )
    valid = depths > 1e-5
    if not np.any(valid):
        return (0, 0, width, height)
    px = pixels[valid, 0]
    py = pixels[valid, 1]
    x0 = max(0, int(np.floor(px.min())) - pad)
    x1 = min(width, int(np.ceil(px.max())) + pad)
    y0 = max(0, int(np.floor(py.min())) - pad)
    y1 = min(height, int(np.ceil(py.max())) + pad)
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return (x0, y0, x1, y1)


def _rasterize_particle_scalar_field(
    pixels: np.ndarray,
    depths: np.ndarray,
    normalized: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    scalar_layer = np.zeros((height, width), dtype=np.float32)
    weight_layer = np.zeros((height, width), dtype=np.float32)
    radius = max(5, int(round(min(width, height) / 140.0)))
    for particle_index in np.argsort(depths)[::-1]:
        if depths[particle_index] <= 1e-5:
            continue

        x_value, y_value = np.rint(pixels[particle_index]).astype(np.int32)
        if (
            x_value < -radius
            or x_value >= width + radius
            or y_value < -radius
            or y_value >= height + radius
        ):
            continue

        center = (int(x_value), int(y_value))
        cv2.circle(
            scalar_layer, center, radius, float(normalized[particle_index]), -1
        )
        cv2.circle(weight_layer, center, radius, 1.0, -1)

    return scalar_layer, weight_layer


def _smooth_scalar_field(
    scalar_layer: np.ndarray,
    weight_layer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = scalar_layer.shape
    radius = max(5, int(round(min(width, height) / 140.0)))
    sigma = max(1.2, float(radius) * 0.55)
    scalar_blur = cv2.GaussianBlur(scalar_layer * weight_layer, (0, 0), sigma)
    weight_blur = cv2.GaussianBlur(weight_layer, (0, 0), sigma)
    scalar_blur = scalar_blur / np.maximum(weight_blur, 1e-6)
    hard_mask = _close_mask(weight_layer > 0.0, radius=max(4, radius * 2))
    scalar_blur[~hard_mask] = 0.0
    weight_blur[~hard_mask] = 0.0
    return scalar_blur, weight_blur


def _close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = max(3, radius * 2 + 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def _compose_rgb_with_inset(
    frame_rgb: np.ndarray,
    scalar_field: np.ndarray,
    weight_field: np.ndarray,
    scale: float,
    inset_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    x0, y0, x1, y1 = inset_bbox

    heat_bgr = cv2.applyColorMap(
        (np.clip(scalar_field, 0.0, 1.0) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat_layer = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    hard_mask = weight_field > 0.0
    alpha = np.where(hard_mask, 0.82, 0.0).astype(np.float32)
    inset = (
        gray_rgb.astype(np.float32) * 0.68
        + frame_rgb.astype(np.float32) * 0.12
    )
    inset = (
        inset * (1.0 - alpha[:, :, None])
        + heat_layer.astype(np.float32) * alpha[:, :, None]
    )
    inset = np.clip(inset[y0:y1, x0:x1], 0, 255).astype(np.uint8)

    inset_width = 250
    crop_height, crop_width = inset.shape[:2]
    inset_height = max(1, int(crop_height * inset_width / max(crop_width, 1)))
    inset = cv2.resize(
        inset, (inset_width, inset_height), interpolation=cv2.INTER_AREA
    )

    legend_width = 30
    legend_gap = 10
    label_width = 54
    margin = 28
    panel_pad = 12
    panel_width = (
        inset_width + legend_gap + legend_width + label_width + panel_pad * 2
    )
    panel_height = inset_height + panel_pad * 2
    panel_x = width - panel_width - margin
    panel_y = margin

    canvas = frame_rgb.copy()
    panel = np.full((panel_height, panel_width, 3), 247, dtype=np.uint8)
    panel = (panel.astype(np.float32) * 0.92).astype(np.uint8)
    cv2.rectangle(
        panel,
        (0, 0),
        (panel_width - 1, panel_height - 1),
        (160, 168, 178),
        1,
    )

    ix = panel_pad
    iy = panel_pad
    panel[iy : iy + inset_height, ix : ix + inset_width] = inset

    bar_x = ix + inset_width + legend_gap
    bar = np.linspace(1.0, 0.0, inset_height, dtype=np.float32)[:, None]
    bar_bgr = cv2.applyColorMap(
        (bar * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)
    panel[iy : iy + inset_height, bar_x : bar_x + legend_width] = np.repeat(
        bar_rgb,
        legend_width,
        axis=1,
    )
    text_x = bar_x + legend_width + 4
    cv2.putText(
        panel,
        f"{scale:.3f}m",
        (text_x, iy + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (35, 40, 48),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "0",
        (text_x, iy + inset_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (35, 40, 48),
        1,
        cv2.LINE_AA,
    )
    canvas[
        panel_y : panel_y + panel_height, panel_x : panel_x + panel_width
    ] = panel
    return canvas


def _project_points(
    points: np.ndarray,
    camera_pos: np.ndarray,
    camera_lookat: np.ndarray,
    camera_fov: float,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    forward = _normalize(camera_lookat - camera_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))

    relative = points - camera_pos[None, :]
    cam_x = relative @ right
    cam_y = relative @ up
    cam_z = relative @ forward
    focal = 0.5 * float(height) / math.tan(math.radians(camera_fov) * 0.5)
    pixels = np.stack(
        [
            width * 0.5 + focal * cam_x / np.maximum(cam_z, 1e-6),
            height * 0.5 - focal * cam_y / np.maximum(cam_z, 1e-6),
        ],
        axis=1,
    )
    return pixels, cam_z


def _draw_legend(image: np.ndarray, scale: float) -> None:
    height, width = image.shape[:2]
    bar_height = min(240, max(120, height // 4))
    bar_width = 26
    margin = 28
    x0 = width - margin - bar_width - 52
    y0 = margin

    panel = image[
        y0 - 10 : y0 + bar_height + 18, x0 - 10 : x0 + bar_width + 52
    ]
    panel[:] = (panel.astype(np.float32) * 0.35 + 245.0 * 0.65).astype(
        np.uint8
    )
    cv2.rectangle(
        image,
        (x0 - 10, y0 - 10),
        (x0 + bar_width + 52, y0 + bar_height + 18),
        (155, 164, 176),
        1,
    )

    values = np.linspace(1.0, 0.0, bar_height, dtype=np.float32)[:, None]
    colors_bgr = cv2.applyColorMap(
        (values * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colors_rgb = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)
    image[y0 : y0 + bar_height, x0 : x0 + bar_width] = np.repeat(
        colors_rgb,
        bar_width,
        axis=1,
    )
    cv2.putText(
        image,
        f"{scale:.3f}m",
        (x0 + bar_width + 6, y0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (35, 40, 48),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "0",
        (x0 + bar_width + 6, y0 + bar_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (35, 40, 48),
        1,
        cv2.LINE_AA,
    )


def _load_obj_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for line in mesh_path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            _, x_value, y_value, z_value, *_ = line.split()
            vertices.append([float(x_value), float(y_value), float(z_value)])
            continue
        if not line.startswith("f "):
            continue

        indices = [
            _parse_obj_index(token, len(vertices))
            for token in line.split()[1:]
        ]
        if len(indices) < 3:
            continue
        for offset in range(1, len(indices) - 1):
            triangles.append(
                [indices[0], indices[offset], indices[offset + 1]]
            )

    if not vertices:
        raise ValueError(f"No vertices found in OBJ mesh: {mesh_path}")
    if not triangles:
        raise ValueError(f"No faces found in OBJ mesh: {mesh_path}")
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
    )


def _map_mesh_faces_to_particles(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    rest_positions: np.ndarray,
    init_height: float,
    mesh_scale: float,
) -> np.ndarray:
    world_vertices = mesh_vertices * float(mesh_scale)
    world_vertices[:, 2] += float(init_height)
    vertex_to_particle = _nearest_indices(
        queries=world_vertices,
        points=rest_positions,
    )
    particle_faces = vertex_to_particle[mesh_faces]
    unique_counts = np.asarray(
        [len(set(face.tolist())) for face in particle_faces],
        dtype=np.int32,
    )
    particle_faces = particle_faces[unique_counts == 3]
    if particle_faces.size == 0:
        raise ValueError(
            "No non-degenerate particle faces remained after mapping"
        )
    return np.unique(particle_faces, axis=0).astype(np.int32)


def _nearest_indices(
    queries: np.ndarray,
    points: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    nearest: list[np.ndarray] = []
    points_sq = np.sum(points * points, axis=1)[None, :]
    for start in range(0, queries.shape[0], chunk_size):
        chunk = queries[start : start + chunk_size]
        distances = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + points_sq
            - 2.0 * chunk @ points.T
        )
        nearest.append(np.argmin(distances, axis=1).astype(np.int32))
    return np.concatenate(nearest, axis=0)


def _parse_obj_index(token: str, vertex_count: int) -> int:
    raw = int(token.split("/")[0])
    if raw < 0:
        return vertex_count + raw
    return raw - 1


def _is_triangle_outside_image(
    triangle: np.ndarray,
    width: int,
    height: int,
) -> bool:
    return (
        triangle[:, 0].max() < 0
        or triangle[:, 0].min() >= width
        or triangle[:, 1].max() < 0
        or triangle[:, 1].min() >= height
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


if __name__ == "__main__":
    main()
