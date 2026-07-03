import os
import sys
import textwrap

import imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from embodied_gen.utils.log import logger

__all__ = [
    "PALETTE",
    "collect_colors",
    "render_grid",
    "visualize_partsemantics",
    "show_grasps",
]

PALETTE = [
    (0, [230, 25, 75], "Red"),
    (1, [60, 180, 75], "Green"),
    (2, [0, 130, 200], "Blue"),
    (3, [255, 225, 25], "Yellow"),
    (4, [245, 130, 48], "Orange"),
    (5, [145, 30, 180], "Purple"),
    (6, [70, 240, 240], "Cyan"),
    (7, [240, 50, 230], "Magenta"),
    (8, [250, 190, 212], "Pink"),
    (9, [210, 245, 60], "Lime Green"),
    (10, [0, 128, 128], "Teal"),
    (11, [170, 110, 40], "Brown"),
    (12, [128, 0, 0], "Maroon"),
    (13, [0, 0, 128], "Navy"),
    (14, [107, 142, 35], "Olive"),
    (15, [128, 128, 128], "Gray"),
    (16, [220, 20, 60], "Crimson"),
    (17, [255, 255, 255], "White"),
    (18, [204, 85, 0], "Burnt Orange"),
    (19, [0, 153, 143], "Jade"),
    (20, [128, 0, 128], "Violet"),
    (21, [255, 215, 0], "Gold"),
    (22, [0, 191, 255], "Deep Sky Blue"),
    (23, [255, 105, 180], "Hot Pink"),
    (24, [124, 252, 0], "Lawn Green"),
    (25, [75, 0, 130], "Indigo"),
    (26, [255, 140, 0], "Dark Orange"),
    (27, [46, 139, 87], "Sea Green"),
    (28, [220, 220, 220], "Gainsboro"),
    (29, [139, 0, 0], "Dark Red"),
    (30, [0, 100, 0], "Dark Green"),
    (31, [25, 25, 112], "Midnight Blue"),
    (32, [255, 20, 147], "Deep Pink"),
    (33, [154, 205, 50], "Yellow Green"),
    (34, [72, 61, 139], "Dark Slate Blue"),
    (35, [255, 160, 122], "Light Salmon"),
    (36, [32, 178, 170], "Light Sea Green"),
    (37, [218, 112, 214], "Orchid"),
    (38, [176, 196, 222], "Light Steel Blue"),
    (39, [189, 183, 107], "Dark Khaki"),
    (40, [255, 99, 71], "Tomato"),
    (41, [64, 224, 208], "Turquoise"),
    (42, [148, 0, 211], "Dark Violet"),
    (43, [240, 128, 128], "Light Coral"),
    (44, [60, 179, 113], "Medium Sea Green"),
    (45, [123, 104, 238], "Medium Slate Blue"),
    (46, [255, 228, 181], "Moccasin"),
    (47, [199, 21, 133], "Medium Violet Red"),
    (48, [112, 128, 144], "Slate Gray"),
    (49, [127, 255, 212], "Aquamarine"),
    (50, [255, 69, 0], "Orange Red"),
    (51, [95, 158, 160], "Cadet Blue"),
    (52, [173, 255, 47], "Green Yellow"),
    (53, [186, 85, 211], "Medium Orchid"),
    (54, [205, 92, 92], "Indian Red"),
    (55, [0, 206, 209], "Dark Turquoise"),
    (56, [255, 182, 193], "Light Pink"),
    (57, [85, 107, 47], "Dark Olive Green"),
    (58, [30, 144, 255], "Dodger Blue"),
    (59, [210, 105, 30], "Chocolate"),
]


def collect_colors(face_ids: np.ndarray) -> str:
    active_ids = {
        int(part_id)
        for part_id in np.unique(np.asarray(face_ids))
        if int(part_id) >= 0
    }
    ordered_colors = [
        color_name
        for palette_id, _, color_name in PALETTE
        if palette_id in active_ids
    ]
    return ", ".join(ordered_colors)


def show_grasps(
    mesh: trimesh.Trimesh,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str,
    model_name: str,
    *,
    grasp_center_points: np.ndarray | None = None,
    visualizer=None,
    visualizer_port: int = 8080,
    gripper_info=None,
    gripper_mesh: trimesh.Trimesh | None = None,
):
    if model_name == "graspgen":
        sys.path.append("thirdparty/GraspGen")
        from grasp_gen.utils.viser_utils import (
            create_visualizer,
            get_color_from_score,
            visualize_grasp,
            visualize_mesh,
            visualize_pointcloud,
        )
    elif model_name == "graspgenx":
        sys.path.append("thirdparty/GraspGenX")
        from graspgenx.utils.viser_utils import (
            create_visualizer,
            get_color_from_score,
            visualize_mesh,
            visualize_pointcloud,
        )
        from graspgenx.utils.viser_utils import (
            visualize_x_grasp as visualize_grasp,
        )

    best_idx = int(confidences.argmax())

    if visualizer is None:
        visualizer = create_visualizer(port=visualizer_port)
    else:
        visualizer.scene.reset()

    transform = np.eye(4, dtype=np.float64)

    if grasp_center_points is not None:
        grasp_point_colors = np.zeros(
            (len(grasp_center_points), 3), dtype=np.uint8
        )
        visualize_pointcloud(
            visualizer,
            "grasp_center_points",
            grasp_center_points,
            grasp_point_colors,
            transform=transform,
            size=0.0025,
        )

    visualize_mesh(
        visualizer,
        "object_mesh",
        mesh,
        color=[169, 169, 169],
        transform=transform,
    )

    colors = get_color_from_score(confidences, use_255_scale=True)
    for idx, grasp in enumerate(grasps):
        color = [0, 100, 255] if idx == best_idx else colors[idx]
        lw = 5.0 if idx == best_idx else 3.0
        if model_name == "graspgen":
            visualize_grasp(
                visualizer,
                f"grasps/grasp_{idx:03d}",
                grasp,
                color=color,
                gripper_name=gripper_name,
                linewidth=lw,
            )
        if model_name == "graspgenx":
            visualize_grasp(
                visualizer,
                f"grasps/grasp_{idx:03d}",
                grasp,
                color=color,
                gripper_info=gripper_info,
                linewidth=lw,
            )

    if gripper_mesh is not None:
        visualize_mesh(
            visualizer,
            "top_grasp_mesh",
            gripper_mesh,
            color=[0, 100, 255],
            transform=grasps[best_idx],
        )

    logger.info(
        "Grasp visualization running at http://localhost:{}".format(
            visualizer_port,
        )
    )
    return visualizer


def render_grid(
    mesh_path: str,
    output_dir: str,
    output_subdir: str = "renders",
    *,
    num_images: int = 6,
    grid_rows: int = 2,
    grid_cols: int = 3,
    view_size: int = 512,
) -> tuple[str, list[str]]:
    from embodied_gen.utils.process_media import (
        combine_images_to_grid,
        render_asset3d,
    )

    image_paths = render_asset3d(
        mesh_path,
        output_root=output_dir,
        distance=5.0,
        num_images=num_images,
        elevation=(45.0, -45.0),
        output_subdir=output_subdir,
        pbr_light_factor=1.5,
        pbr_metallic=False,
        no_index_file=True,
    )
    image_paths = sorted(image_paths)
    if len(image_paths) != num_images:
        raise ValueError(
            "grid rendering expected "
            f"{num_images} images, got {len(image_paths)}."
        )

    grid = combine_images_to_grid(
        image_paths,
        cat_row_col=(grid_rows, grid_cols),
        target_wh=(view_size, view_size),
        image_mode="RGB",
    )[0]
    grid_path = os.path.join(output_dir, output_subdir, "affordance_grid.png")
    grid.save(grid_path)
    return grid_path, image_paths


def _build_fixed_camera_projection(
    vertices: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if extent <= 0:
        extent = 1.0

    view_dir = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
    view_dir /= np.linalg.norm(view_dir)
    eye = center + view_dir * extent * 1.8
    forward = center - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    right, up = -up, right

    relative_vertices = vertices - eye
    camera_points = np.stack(
        [
            relative_vertices @ right,
            relative_vertices @ up,
            relative_vertices @ forward,
        ],
        axis=1,
    )
    depth = np.maximum(camera_points[:, 2], 1e-6)
    focal = image_size * 1.35
    projected = np.empty((len(vertices), 2), dtype=np.float64)
    projected[:, 0] = image_size * 0.5 + focal * camera_points[:, 0] / depth
    projected[:, 1] = image_size * 0.5 - focal * camera_points[:, 1] / depth

    xy_min = projected.min(axis=0)
    xy_max = projected.max(axis=0)
    projected_center = (xy_min + xy_max) / 2.0
    projected_extent = float(np.max(xy_max - xy_min))
    if projected_extent > 1e-6:
        scale = image_size * 0.82 / projected_extent
        projected = (projected - projected_center) * scale + image_size * 0.5
    return projected, camera_points[:, 2]


def _build_affordance_text_lines(affordance: dict) -> list[str]:
    title = f"{affordance.get('part_name', 'unknown part')}  id={affordance.get('id')}"
    lines = [
        title,
        f"mask color: {affordance.get('mask_color')}",
        f"graspable: {affordance.get('graspable')}",
    ]

    scenarios = affordance.get("grasp_scenarios") or []
    if scenarios:
        lines.append("grasp scenarios:")
        for item in scenarios[:3]:
            scenario = item.get("scenario", "")
            confidence = item.get("confidence")
            lines.append(f"- {scenario} ({confidence})")

    labels = affordance.get("functional_labels") or []
    if labels:
        lines.append("functions:")
        lines.extend([f"- {label}" for label in labels[:6]])

    semantic = affordance.get("semantic_description")
    if semantic:
        lines.append("semantic:")
        lines.extend(textwrap.wrap(str(semantic), width=42))
    return lines


def _load_visualization_fonts(
    image_size: int,
) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    scale = max(1.0, image_size / 512)
    title_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    ]
    body_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    ]

    def _load_font(font_paths: list[str], size: int) -> ImageFont.ImageFont:
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size=size)
        return ImageFont.load_default()

    return (
        _load_font(title_font_paths, int(24 * scale)),
        _load_font(body_font_paths, int(18 * scale)),
    )


def _draw_affordance_text_panel(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    panel_width: int,
    height: int,
    lines: list[str],
    visible_chars: int,
) -> None:
    title_font, body_font = _load_visualization_fonts(height)
    scale = max(1.0, height / 512)
    left_margin = int(24 * scale)
    title_line_height = int(30 * scale)
    body_line_height = int(24 * scale)
    draw.rectangle(
        [panel_x, 0, panel_x + panel_width, height], fill=(248, 248, 240)
    )
    draw.rectangle(
        [panel_x, 0, panel_x + int(1 * scale), height], fill=(0, 105, 160)
    )
    y = int(28 * scale)
    remaining_chars = visible_chars
    for idx, line in enumerate(lines):
        font = title_font if idx == 0 else body_font
        line_height = title_line_height if idx == 0 else body_line_height
        wrap_width = 30 if idx == 0 else 38
        wrapped_lines = textwrap.wrap(str(line), width=wrap_width) or [""]
        for wrapped in wrapped_lines:
            if remaining_chars <= 0:
                return
            visible_text = wrapped[:remaining_chars]
            fill = (28, 31, 34) if idx != 0 else (0, 82, 125)
            draw.text(
                (panel_x + left_margin, y), visible_text, fill=fill, font=font
            )
            remaining_chars -= len(wrapped)
            y += line_height
        y += int(10 * scale) if idx == 0 else int(7 * scale)


def _draw_affordance_frame(
    mesh: trimesh.Trimesh,
    face_part_ids: np.ndarray,
    face_colors: np.ndarray,
    screen_vertices: np.ndarray,
    vertex_depths: np.ndarray,
    affordance: dict,
    blink_alpha: int,
    text_lines: list[str],
    visible_chars: int,
    image_size: int,
    text_panel_width: int,
) -> Image.Image:
    canvas = Image.new(
        "RGBA",
        (image_size + text_panel_width, image_size),
        (255, 255, 255, 255),
    )
    mesh_layer = Image.new(
        "RGBA", (image_size, image_size), (255, 255, 255, 0)
    )
    base_draw = ImageDraw.Draw(mesh_layer, "RGBA")
    highlight_layer = Image.new(
        "RGBA", (image_size, image_size), (255, 255, 255, 0)
    )
    highlight_draw = ImageDraw.Draw(highlight_layer, "RGBA")

    faces = np.asarray(mesh.faces)
    face_depths = vertex_depths[faces].mean(axis=1)
    draw_order = np.argsort(face_depths)[::-1]
    target_part_id = affordance.get("id")
    highlight_rgb = tuple(
        next(
            (
                color
                for palette_id, color, _ in PALETTE
                if palette_id == target_part_id
            ),
            [255, 40, 40],
        )
    )
    bright_highlight_rgb = tuple(
        min(255, int(channel * 1.35 + 35)) for channel in highlight_rgb
    )

    for face_idx in draw_order:
        polygon = [
            tuple(screen_vertices[vertex_idx])
            for vertex_idx in faces[face_idx]
        ]
        if face_part_ids[face_idx] == target_part_id:
            fill = (*bright_highlight_rgb, blink_alpha)
            outline = (255, 255, 255, 255)
            highlight_draw.polygon(polygon, fill=fill, outline=outline)
        else:
            base_rgb = tuple(
                max(0, int(channel * 0.58))
                for channel in face_colors[face_idx][:3]
            )
            base_draw.polygon(
                polygon,
                fill=(*base_rgb, 48),
                outline=(*base_rgb, 12),
            )

    canvas.alpha_composite(mesh_layer, (0, 0))
    canvas.alpha_composite(highlight_layer, (0, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_affordance_text_panel(
        draw,
        image_size,
        text_panel_width,
        image_size,
        text_lines,
        visible_chars,
    )
    return canvas.convert("RGB")


def _get_face_base_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    fallback = np.full((len(mesh.faces), 3), 170, dtype=np.uint8)
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return fallback

    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None and len(face_colors) == len(mesh.faces):
        return np.asarray(face_colors, dtype=np.uint8)[:, :3]

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        vertex_colors = np.asarray(vertex_colors, dtype=np.float32)[:, :3]
        return (
            vertex_colors[np.asarray(mesh.faces)].mean(axis=1).astype(np.uint8)
        )

    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    uv = getattr(visual, "uv", None)
    if (
        texture is not None
        and uv is not None
        and len(uv) == len(mesh.vertices)
    ):
        texture_arr = np.asarray(texture.convert("RGB"), dtype=np.uint8)
        height, width = texture_arr.shape[:2]
        face_uv = np.asarray(uv, dtype=np.float64)[
            np.asarray(mesh.faces)
        ].mean(axis=1)
        u = np.mod(face_uv[:, 0], 1.0)
        v = np.mod(face_uv[:, 1], 1.0)
        x = np.clip((u * (width - 1)).round().astype(np.int64), 0, width - 1)
        y = np.clip(
            ((1.0 - v) * (height - 1)).round().astype(np.int64), 0, height - 1
        )
        return texture_arr[y, x, :3]

    diffuse = getattr(material, "diffuse", None)
    if diffuse is not None and len(diffuse) >= 3:
        color = np.asarray(diffuse[:3], dtype=np.uint8)
        return np.tile(color, (len(mesh.faces), 1))

    base_color = getattr(material, "baseColorFactor", None)
    if base_color is not None and len(base_color) >= 3:
        color = np.asarray(base_color[:3], dtype=np.uint8)
        return np.tile(color, (len(mesh.faces), 1))

    return fallback


def visualize_partsemantics(
    mesh_path: str,
    seg_mesh_path: str,
    affordance_annot_path: str,
    *,
    fps: int = 12,
    frames_per_part: int = 36,
    view_size: int = 1080,
    output_path: str | None = None,
) -> str | None:

    from embodied_gen.utils.io_utils import load_json

    affordance_payload = load_json(affordance_annot_path)
    affordances = [
        item
        for item in affordance_payload.get("affordances", [])
        if isinstance(item, dict) and item.get("id") is not None
    ]
    if not affordances:
        logger.warning(
            f"No affordance entries to visualize: {affordance_annot_path}"
        )
        return None

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Expected mesh at {mesh_path}, got {type(mesh).__name__}"
        )

    face_colors = _get_face_base_colors(mesh)
    seg_mesh = trimesh.load(seg_mesh_path, force="mesh", process=False)
    face_part_ids = np.asarray(seg_mesh.metadata["face_ids"], dtype=np.int64)
    if len(face_part_ids) != len(mesh.faces):
        raise ValueError(
            "face_ids length must match mesh faces, got "
            f"{len(face_part_ids)} and {len(mesh.faces)}"
        )

    palette_ids = {palette_id for palette_id, _, _ in PALETTE}
    part_ids = np.unique(face_part_ids[face_part_ids >= 0])
    if any(int(part_id) not in palette_ids for part_id in part_ids):
        if len(part_ids) > len(PALETTE):
            raise ValueError(
                "number of unique part ids "
                f"{len(part_ids)} exceeds color palette size {len(PALETTE)}"
            )
        palette_part_ids = np.full(face_part_ids.shape, -1, dtype=np.int64)
        for palette_idx, part_id in enumerate(part_ids):
            palette_part_ids[face_part_ids == part_id] = PALETTE[palette_idx][
                0
            ]
        face_part_ids = palette_part_ids

    image_size = view_size
    text_panel_width = int(view_size * 0.75)
    screen_vertices, vertex_depths = _build_fixed_camera_projection(
        np.asarray(mesh.vertices, dtype=np.float64),
        image_size,
    )

    frames = []
    frames_per_affordance = max(1, frames_per_part)
    for affordance in affordances:
        text_lines = _build_affordance_text_lines(affordance)
        total_chars = sum(len(line) for line in text_lines)
        for frame_idx in range(frames_per_affordance):
            phase = frame_idx / max(1, frames_per_affordance - 1)
            blink_strength = 1.0 - abs(2.0 * phase - 1.0)
            blink_alpha = int(130 + 125 * blink_strength)
            visible_chars = int(total_chars * min(1.0, phase * 1.25))
            frame = _draw_affordance_frame(
                mesh,
                face_part_ids,
                face_colors,
                screen_vertices,
                vertex_depths,
                affordance,
                blink_alpha,
                text_lines,
                visible_chars,
                image_size,
                text_panel_width,
            )
            frames.append(np.asarray(frame))

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(affordance_annot_path),
            "affordance_annot_vis.mp4",
        )
    imageio.mimsave(output_path, frames, fps=fps)
    logger.info(f"Affordance visualization saved to {output_path}")
    return output_path
