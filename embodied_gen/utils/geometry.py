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
import random
from collections import defaultdict, deque
from functools import wraps
from typing import Literal

import numpy as np
import torch
import trimesh
from matplotlib.path import Path
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.log import logger

__all__ = [
    "with_seed",
    "matrix_to_pose",
    "pose_to_matrix",
    "quaternion_multiply",
    "check_reachable",
    "bfs_placement",
    "compose_mesh_scene",
    "compute_pinhole_intrinsics",
]


def matrix_to_pose(matrix: np.ndarray) -> list[float]:
    """Convert a 4x4 transformation matrix to a pose (x, y, z, qx, qy, qz, qw).

    Args:
        matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        List[float]: Pose as [x, y, z, qx, qy, qz, qw].
    """
    x, y, z = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    quat = R.from_matrix(rot_mat).as_quat()
    qx, qy, qz, qw = quat

    return [x, y, z, qx, qy, qz, qw]


def pose_to_matrix(pose: list[float]) -> np.ndarray:
    """Convert pose (x, y, z, qx, qy, qz, qw) to a 4x4 transformation matrix.

    Args:
        List[float]: Pose as [x, y, z, qx, qy, qz, qw].

    Returns:
        matrix (np.ndarray): 4x4 transformation matrix.
    """
    x, y, z, qx, qy, qz, qw = pose
    r = R.from_quat([qx, qy, qz, qw])
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    matrix[:3, 3] = [x, y, z]

    return matrix


def compute_xy_bbox(
    vertices: np.ndarray, col_x: int = 0, col_y: int = 2
) -> list[float]:
    x_vals = vertices[:, col_x]
    y_vals = vertices[:, col_y]
    return x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()


def has_iou_conflict(
    new_box: list[float],
    placed_boxes: list[list[float]],
    iou_threshold: float = 0.0,
) -> bool:
    new_min_x, new_max_x, new_min_y, new_max_y = new_box
    for min_x, max_x, min_y, max_y in placed_boxes:
        ix1 = max(new_min_x, min_x)
        iy1 = max(new_min_y, min_y)
        ix2 = min(new_max_x, max_x)
        iy2 = min(new_max_y, max_y)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter_area > iou_threshold:
            return True
    return False


def with_seed(seed_attr_name: str = "seed"):
    """A parameterized decorator that temporarily sets the random seed."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            seed = kwargs.get(seed_attr_name, None)
            if seed is not None:
                py_state = random.getstate()
                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                try:
                    result = func(*args, **kwargs)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    torch.set_rng_state(torch_state)
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def compute_convex_hull_path(
    vertices: np.ndarray,
    z_threshold: float = 0.05,
    interp_per_edge: int = 3,
    margin: float = -0.02,
) -> Path:
    top_vertices = vertices[
        vertices[:, 1] > vertices[:, 1].max() - z_threshold
    ]
    top_xy = top_vertices[:, [0, 2]]

    if len(top_xy) < 3:
        raise ValueError("Not enough points to form a convex hull")

    hull = ConvexHull(top_xy)
    hull_points = top_xy[hull.vertices]

    polygon = Polygon(hull_points)
    polygon = polygon.buffer(margin)
    hull_points = np.array(polygon.exterior.coords)

    dense_points = []
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        for t in np.linspace(0, 1, interp_per_edge, endpoint=False):
            pt = (1 - t) * p1 + t * p2
            dense_points.append(pt)

    return Path(np.array(dense_points), closed=True)


def find_parent_node(node: str, tree: dict) -> str | None:
    for parent, children in tree.items():
        if any(child[0] == node for child in children):
            return parent
    return None


def all_corners_inside(hull: Path, box: list, threshold: int = 3) -> bool:
    x1, x2, y1, y2 = box
    corners = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]

    num_inside = sum(hull.contains_point(c) for c in corners)
    return num_inside >= threshold


def compute_axis_rotation_quat(
    axis: Literal["x", "y", "z"], angle_rad: float
) -> list[float]:
    if axis.lower() == 'x':
        q = Quaternion(axis=[1, 0, 0], angle=angle_rad)
    elif axis.lower() == 'y':
        q = Quaternion(axis=[0, 1, 0], angle=angle_rad)
    elif axis.lower() == 'z':
        q = Quaternion(axis=[0, 0, 1], angle=angle_rad)
    else:
        raise ValueError(f"Unsupported axis '{axis}', must be one of x, y, z")

    return [q.x, q.y, q.z, q.w]


def quaternion_multiply(
    init_quat: list[float], rotate_quat: list[float]
) -> list[float]:
    qx, qy, qz, qw = init_quat
    q1 = Quaternion(w=qw, x=qx, y=qy, z=qz)
    qx, qy, qz, qw = rotate_quat
    q2 = Quaternion(w=qw, x=qx, y=qy, z=qz)
    quat = q2 * q1

    return [quat.x, quat.y, quat.z, quat.w]


def check_reachable(
    base_xyz: np.ndarray,
    reach_xyz: np.ndarray,
    min_reach: float = 0.25,
    max_reach: float = 0.85,
) -> bool:
    """Check if the target point is within the reachable range."""
    distance = np.linalg.norm(reach_xyz - base_xyz)

    return min_reach < distance < max_reach


@with_seed("seed")
def bfs_placement(
    layout_file: str,
    floor_margin: float = 0,
    beside_margin: float = 0.1,
    max_attempts: int = 3000,
    rotate_objs: bool = True,
    rotate_bg: bool = True,
    limit_reach_range: bool = True,
    robot_dim: float = 0.12,
    seed: int = None,
) -> LayoutInfo:
    layout_info = LayoutInfo.from_dict(json.load(open(layout_file, "r")))
    asset_dir = os.path.dirname(layout_file)
    object_mapping = layout_info.objs_mapping
    position = {}  # node: [x, y, z, qx, qy, qz, qw]
    parent_bbox_xy = {}
    placed_boxes_map = defaultdict(list)
    mesh_info = defaultdict(dict)
    robot_node = layout_info.relation[Scene3DItemEnum.ROBOT.value]
    for node in object_mapping:
        if object_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
            bg_quat = (
                compute_axis_rotation_quat(
                    axis="y",
                    angle_rad=np.random.uniform(0, 2 * np.pi),
                )
                if rotate_bg
                else [0, 0, 0, 1]
            )
            bg_quat = [round(q, 4) for q in bg_quat]
            continue

        mesh_path = (
            f"{layout_info.assets[node]}/mesh/{node.replace(' ', '_')}.obj"
        )
        mesh_path = os.path.join(asset_dir, mesh_path)
        mesh_info[node]["path"] = mesh_path
        mesh = trimesh.load(mesh_path)
        vertices = mesh.vertices
        z1 = np.percentile(vertices[:, 1], 1)
        z2 = np.percentile(vertices[:, 1], 99)

        if object_mapping[node] == Scene3DItemEnum.CONTEXT.value:
            object_quat = [0, 0, 0, 1]
            mesh_info[node]["surface"] = compute_convex_hull_path(vertices)
            # Put robot in the CONTEXT edge.
            x, y = random.choice(mesh_info[node]["surface"].vertices)
            theta = np.arctan2(y, x)
            quat_initial = Quaternion(axis=[0, 0, 1], angle=theta)
            quat_extra = Quaternion(axis=[0, 0, 1], angle=np.pi)
            quat = quat_extra * quat_initial
            _pose = [x, y, z2 - z1, quat.x, quat.y, quat.z, quat.w]
            position[robot_node] = [round(v, 4) for v in _pose]
            node_box = [
                x - robot_dim / 2,
                x + robot_dim / 2,
                y - robot_dim / 2,
                y + robot_dim / 2,
            ]
            placed_boxes_map[node].append(node_box)
        elif rotate_objs:
            # For manipulated and distractor objects, apply random rotation
            angle_rad = np.random.uniform(0, 2 * np.pi)
            object_quat = compute_axis_rotation_quat(
                axis="z", angle_rad=angle_rad
            )
            rotation = R.from_quat(object_quat).as_matrix()
            vertices = np.dot(mesh.vertices, rotation.T)
            z1 = np.percentile(vertices[:, 1], 1)
            z2 = np.percentile(vertices[:, 1], 99)

        x1, x2, y1, y2 = compute_xy_bbox(vertices)
        mesh_info[node]["pose"] = [x1, x2, y1, y2, z1, z2, *object_quat]
        mesh_info[node]["area"] = max(1e-5, (x2 - x1) * (y2 - y1))

    root = list(layout_info.tree.keys())[0]
    queue = deque([((root, None), layout_info.tree.get(root, []))])
    while queue:
        (node, relation), children = queue.popleft()
        if node not in object_mapping:
            continue

        if object_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
            position[node] = [0, 0, floor_margin, *bg_quat]
        else:
            x1, x2, y1, y2, z1, z2, qx, qy, qz, qw = mesh_info[node]["pose"]
            if object_mapping[node] == Scene3DItemEnum.CONTEXT.value:
                position[node] = [0, 0, -round(z1, 4), qx, qy, qz, qw]
                parent_bbox_xy[node] = [x1, x2, y1, y2, z1, z2]
            elif object_mapping[node] in [
                Scene3DItemEnum.MANIPULATED_OBJS.value,
                Scene3DItemEnum.DISTRACTOR_OBJS.value,
            ]:
                parent_node = find_parent_node(node, layout_info.tree)
                parent_pos = position[parent_node]
                (
                    p_x1,
                    p_x2,
                    p_y1,
                    p_y2,
                    p_z1,
                    p_z2,
                ) = parent_bbox_xy[parent_node]

                obj_dx = x2 - x1
                obj_dy = y2 - y1
                hull_path = mesh_info[parent_node].get("surface")
                for _ in range(max_attempts):
                    node_x1 = random.uniform(p_x1, p_x2 - obj_dx)
                    node_y1 = random.uniform(p_y1, p_y2 - obj_dy)
                    node_box = [
                        node_x1,
                        node_x1 + obj_dx,
                        node_y1,
                        node_y1 + obj_dy,
                    ]
                    if hull_path and not all_corners_inside(
                        hull_path, node_box
                    ):
                        continue
                    # Make sure the manipulated object is reachable by robot.
                    if (
                        limit_reach_range
                        and object_mapping[node]
                        == Scene3DItemEnum.MANIPULATED_OBJS.value
                    ):
                        cx = parent_pos[0] + node_box[0] + obj_dx / 2
                        cy = parent_pos[1] + node_box[2] + obj_dy / 2
                        cz = parent_pos[2] + p_z2 - z1
                        robot_pose = position[robot_node][:3]
                        if not check_reachable(
                            base_xyz=np.array(robot_pose),
                            reach_xyz=np.array([cx, cy, cz]),
                        ):
                            continue

                    if not has_iou_conflict(
                        node_box, placed_boxes_map[parent_node]
                    ):
                        z_offset = 0
                        break
                else:
                    logger.warning(
                        f"Cannot place {node} on {parent_node} without overlap"
                        f" after {max_attempts} attempts, place beside {parent_node}."
                    )
                    for _ in range(max_attempts):
                        node_x1 = random.choice(
                            [
                                random.uniform(
                                    p_x1 - obj_dx - beside_margin,
                                    p_x1 - obj_dx,
                                ),
                                random.uniform(p_x2, p_x2 + beside_margin),
                            ]
                        )
                        node_y1 = random.choice(
                            [
                                random.uniform(
                                    p_y1 - obj_dy - beside_margin,
                                    p_y1 - obj_dy,
                                ),
                                random.uniform(p_y2, p_y2 + beside_margin),
                            ]
                        )
                        node_box = [
                            node_x1,
                            node_x1 + obj_dx,
                            node_y1,
                            node_y1 + obj_dy,
                        ]
                        z_offset = -(parent_pos[2] + p_z2)
                        if not has_iou_conflict(
                            node_box, placed_boxes_map[parent_node]
                        ):
                            break

                placed_boxes_map[parent_node].append(node_box)

                abs_cx = parent_pos[0] + node_box[0] + obj_dx / 2
                abs_cy = parent_pos[1] + node_box[2] + obj_dy / 2
                abs_cz = parent_pos[2] + p_z2 - z1 + z_offset
                position[node] = [
                    round(v, 4)
                    for v in [abs_cx, abs_cy, abs_cz, qx, qy, qz, qw]
                ]
                parent_bbox_xy[node] = [x1, x2, y1, y2, z1, z2]

        sorted_children = sorted(
            children, key=lambda x: -mesh_info[x[0]].get("area", 0)
        )
        for child, rel in sorted_children:
            queue.append(((child, rel), layout_info.tree.get(child, [])))

    layout_info.position = position

    return layout_info


def compose_mesh_scene(
    layout_info: LayoutInfo, out_scene_path: str, with_bg: bool = False
) -> None:
    object_mapping = Scene3DItemEnum.object_mapping(layout_info.relation)
    scene = trimesh.Scene()
    for node in layout_info.assets:
        if object_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
            mesh_path = f"{layout_info.assets[node]}/mesh_model.ply"
            if not with_bg:
                continue
        else:
            mesh_path = (
                f"{layout_info.assets[node]}/mesh/{node.replace(' ', '_')}.obj"
            )

        mesh = trimesh.load(mesh_path)
        offset = np.array(layout_info.position[node])[[0, 2, 1]]
        mesh.vertices += offset
        scene.add_geometry(mesh, node_name=node)

    os.makedirs(os.path.dirname(out_scene_path), exist_ok=True)
    scene.export(out_scene_path)
    logger.info(f"Composed interactive 3D layout saved in {out_scene_path}")

    return


def compute_pinhole_intrinsics(
    image_w: int, image_h: int, fov_deg: float
) -> np.ndarray:
    fov_rad = np.deg2rad(fov_deg)
    fx = image_w / (2 * np.tan(fov_rad / 2))
    fy = fx  # assuming square pixels
    cx = image_w / 2
    cy = image_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K
