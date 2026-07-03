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
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    import sapien.core as sapien

__all__ = [
    "EnclosedFaceLabelFillConfig",
    "MeshInfo",
    "SmoothFaceLabelMergeConfig",
    "with_seed",
    "fill_small_enclosed_face_labels",
    "normalize_mesh",
    "merge_smooth_face_labels",
    "nearest_faces",
    "sample_surface_points",
    "gripper_finger_center",
    "matrix_to_pose",
    "pose_to_matrix",
    "quaternion_multiply",
    "check_reachable",
    "bfs_placement",
    "compose_mesh_scene",
    "compute_pinhole_intrinsics",
]


@dataclass
class MeshInfo:
    actor_name: str
    collision_mesh_path: str
    collision_mesh_scale: np.ndarray
    collision_local_pose: sapien.Pose
    visual_mesh_path: str
    visual_mesh_scale: np.ndarray
    visual_local_pose: sapien.Pose
    transformed_mesh: trimesh.Trimesh
    object_height: float
    mass: float | None
    static_friction: float
    dynamic_friction: float


@dataclass(frozen=True)
class SmoothFaceLabelMergeConfig:
    max_smooth_angle_deg: float = 18.0
    vertex_merge_tolerance: float = 0.0
    min_component_area_ratio: float = 0.0
    min_component_faces: int = 1
    preserve_negative: bool = False


@dataclass(frozen=True)
class EnclosedFaceLabelFillConfig:
    max_component_area_ratio: float = 0.005
    max_component_faces: int = 100
    min_enclosure_neighbor_ratio: float = 0.8
    vertex_merge_tolerance: float = 0.0


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh_unit = mesh.copy()
    scale = float(np.ptp(mesh_unit.vertices, axis=0).max())
    if scale <= 0.0:
        raise ValueError("mesh has zero extent and cannot be normalized")
    mesh_unit.apply_scale(1.0 / scale)
    return mesh_unit


def sample_surface_points(
    mesh: trimesh.Trimesh, num_sample_points: int
) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    points = np.asarray(points, dtype=np.float32)

    if len(points) > num_sample_points:
        indices = np.random.choice(
            len(points), num_sample_points, replace=False
        )
        points = points[indices]

    return points


def nearest_faces(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return np.full(len(points), -1, dtype=np.int64)

    try:
        _, _, face_ids = trimesh.proximity.closest_point(mesh, points)
        return np.asarray(face_ids, dtype=np.int64)
    except Exception as exc:
        logger.warning(
            "Failed to query closest mesh faces with trimesh proximity, "
            f"falling back to triangle-center KDTree: {exc}"
        )
        from scipy.spatial import cKDTree

        _, face_ids = cKDTree(mesh.triangles_center).query(points)
        return np.asarray(face_ids, dtype=np.int64)


def _auto_vertex_merge_tolerance(mesh: trimesh.Trimesh) -> float:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        return 1e-8
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    return max(diag * 1e-8, 1e-10)


def _vertex_key(vertex: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    return tuple(np.rint(vertex / tolerance).astype(np.int64).tolist())


def _edge_face_groups_by_position(
    mesh: trimesh.Trimesh,
    vertex_merge_tolerance: float = 0.0,
) -> dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[int]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(faces) == 0:
        return {}

    tolerance = (
        float(vertex_merge_tolerance)
        if vertex_merge_tolerance > 0.0
        else _auto_vertex_merge_tolerance(mesh)
    )
    edge_to_faces: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]], list[int]
    ] = {}
    edge_offsets = ((0, 1), (1, 2), (2, 0))
    for face_idx, face in enumerate(faces):
        for left_offset, right_offset in edge_offsets:
            left_key = _vertex_key(vertices[face[left_offset]], tolerance)
            right_key = _vertex_key(vertices[face[right_offset]], tolerance)
            edge_key = tuple(sorted((left_key, right_key)))
            edge_to_faces.setdefault(edge_key, []).append(face_idx)
    return edge_to_faces


def _face_adjacency_by_position(
    mesh: trimesh.Trimesh,
    vertex_merge_tolerance: float = 0.0,
) -> np.ndarray:
    edge_to_faces = _edge_face_groups_by_position(mesh, vertex_merge_tolerance)
    adjacent_pairs = []
    for face_indices in edge_to_faces.values():
        if len(face_indices) < 2:
            continue
        for idx in range(len(face_indices) - 1):
            adjacent_pairs.append((face_indices[idx], face_indices[idx + 1]))

    if not adjacent_pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(adjacent_pairs, dtype=np.int64)


def _face_adjacency_lists(
    n_faces: int,
    adjacency: np.ndarray,
) -> list[list[int]]:
    adjacency_lists = [[] for _ in range(n_faces)]
    for left, right in adjacency:
        adjacency_lists[int(left)].append(int(right))
        adjacency_lists[int(right)].append(int(left))
    return adjacency_lists


def _smooth_face_components(
    mesh: trimesh.Trimesh,
    max_smooth_angle_deg: float,
    vertex_merge_tolerance: float = 0.0,
) -> list[np.ndarray]:
    if len(mesh.faces) == 0:
        return []

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    adjacency = _face_adjacency_by_position(mesh, vertex_merge_tolerance)
    union_find = _UnionFind(len(mesh.faces))
    if len(adjacency) > 0:
        normal_dot = np.einsum(
            "ij,ij->i",
            face_normals[adjacency[:, 0]],
            face_normals[adjacency[:, 1]],
        )
        normal_dot = np.clip(normal_dot, -1.0, 1.0)
        angles = np.degrees(np.arccos(normal_dot))
        smooth_pairs = adjacency[angles <= max_smooth_angle_deg]
        for left, right in smooth_pairs:
            union_find.union(int(left), int(right))

    groups: dict[int, list[int]] = {}
    for face_idx in range(len(mesh.faces)):
        groups.setdefault(union_find.find(face_idx), []).append(face_idx)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _same_label_components(
    face_labels: np.ndarray,
    adjacency_lists: list[list[int]],
) -> list[np.ndarray]:
    visited = np.zeros(len(face_labels), dtype=bool)
    components = []
    for face_idx, label in enumerate(face_labels):
        if visited[face_idx] or label < 0:
            continue

        visited[face_idx] = True
        stack = [face_idx]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency_lists[current]:
                if visited[neighbor] or face_labels[neighbor] != label:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def _dominant_face_label(
    face_labels: np.ndarray,
    face_areas: np.ndarray,
    face_indices: np.ndarray,
) -> int | None:
    labels = face_labels[face_indices]
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return None

    valid_labels = labels[valid_mask]
    valid_areas = face_areas[face_indices][valid_mask]
    area_by_label: dict[int, float] = {}
    for label, area in zip(valid_labels, valid_areas):
        area_by_label[int(label)] = area_by_label.get(int(label), 0.0) + float(
            area
        )
    return max(area_by_label.items(), key=lambda item: item[1])[0]


def fill_small_enclosed_face_labels(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    cfg: EnclosedFaceLabelFillConfig | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    if cfg is None:
        cfg = EnclosedFaceLabelFillConfig()
    if len(face_labels) != len(mesh.faces):
        raise ValueError(
            "face_labels length must match mesh faces, got "
            f"{len(face_labels)} and {len(mesh.faces)}"
        )

    new_face_labels = np.asarray(face_labels, dtype=np.int64).copy()
    edge_to_faces = _edge_face_groups_by_position(
        mesh,
        cfg.vertex_merge_tolerance,
    )
    adjacency = _face_adjacency_by_position(mesh, cfg.vertex_merge_tolerance)
    adjacency_lists = _face_adjacency_lists(len(mesh.faces), adjacency)
    open_boundary_faces = {
        face_indices[0]
        for face_indices in edge_to_faces.values()
        if len(face_indices) == 1
    }

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(np.sum(face_areas))
    max_area = max(0.0, cfg.max_component_area_ratio) * total_area
    max_faces = int(cfg.max_component_faces)

    component_mask = np.zeros(len(mesh.faces), dtype=bool)
    filled_components = 0
    skipped_components = 0
    for face_indices in _same_label_components(face_labels, adjacency_lists):
        source_label = int(face_labels[face_indices[0]])
        component_area = float(np.sum(face_areas[face_indices]))
        if max_area <= 0.0 or component_area > max_area:
            skipped_components += 1
            continue
        if max_faces > 0 and len(face_indices) > max_faces:
            skipped_components += 1
            continue
        if any(
            int(face_idx) in open_boundary_faces for face_idx in face_indices
        ):
            skipped_components += 1
            continue

        component_mask[face_indices] = True
        outside_labels = []
        for face_idx in face_indices:
            for neighbor in adjacency_lists[int(face_idx)]:
                if component_mask[neighbor]:
                    continue
                outside_labels.append(int(face_labels[neighbor]))

        target_counts: dict[int, int] = {}
        for label in outside_labels:
            if label >= 0 and label != source_label:
                target_counts[label] = target_counts.get(label, 0) + 1

        target_label = None
        if outside_labels and target_counts:
            target_label, target_count = max(
                target_counts.items(),
                key=lambda item: item[1],
            )
            enclosure_ratio = target_count / len(outside_labels)
            if enclosure_ratio < cfg.min_enclosure_neighbor_ratio:
                target_label = None

        if target_label is None:
            skipped_components += 1
        else:
            new_face_labels[face_indices] = target_label
            filled_components += 1
        component_mask[face_indices] = False

    stats = {
        "components_filled": filled_components,
        "components_skipped": skipped_components,
        "faces_changed": int(np.count_nonzero(new_face_labels != face_labels)),
    }
    return new_face_labels, stats


def merge_smooth_face_labels(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    cfg: SmoothFaceLabelMergeConfig | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    if cfg is None:
        cfg = SmoothFaceLabelMergeConfig()
    if len(face_labels) != len(mesh.faces):
        raise ValueError(
            "face_labels length must match mesh faces, got "
            f"{len(face_labels)} and {len(mesh.faces)}"
        )

    new_face_labels = np.asarray(face_labels, dtype=np.int64).copy()
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(np.sum(face_areas))
    min_area = max(0.0, cfg.min_component_area_ratio) * total_area

    changed_components = 0
    skipped_components = 0
    for face_indices in _smooth_face_components(
        mesh,
        cfg.max_smooth_angle_deg,
        cfg.vertex_merge_tolerance,
    ):
        if len(face_indices) < cfg.min_component_faces:
            skipped_components += 1
            continue
        component_area = float(np.sum(face_areas[face_indices]))
        if component_area < min_area:
            skipped_components += 1
            continue

        labels = new_face_labels[face_indices]
        valid_unique = np.unique(labels[labels >= 0])
        if len(valid_unique) <= 1:
            continue

        target_label = _dominant_face_label(
            new_face_labels,
            face_areas,
            face_indices,
        )
        if target_label is None:
            continue

        if cfg.preserve_negative:
            assign_mask = new_face_labels[face_indices] >= 0
            assign_faces = face_indices[assign_mask]
        else:
            assign_faces = face_indices

        new_face_labels[assign_faces] = target_label
        changed_components += 1

    stats = {
        "components_changed": changed_components,
        "components_skipped": skipped_components,
        "faces_changed": int(np.count_nonzero(new_face_labels != face_labels)),
    }
    return new_face_labels, stats


def gripper_finger_center(
    gripper_urdf_path: str,
    geometry_type: Literal["collision", "visual"] = "collision",
) -> np.ndarray:
    from embodied_gen.utils.io_utils import URDFFile

    gripper_urdf = URDFFile(gripper_urdf_path)
    finger_link_names = gripper_urdf.get_child_link_names(
        name_contains="finger"
    )
    if len(finger_link_names) != 2:
        raise ValueError(
            "expected exactly two finger links in gripper URDF, got "
            f"{finger_link_names}: {gripper_urdf_path}"
        )

    link_transforms = gripper_urdf.get_link_transforms()
    finger_centers = []
    for link_name in finger_link_names:
        mesh = gripper_urdf.load_link_geometry_mesh(link_name, geometry_type)
        mesh.apply_transform(link_transforms[link_name])
        finger_centers.append(np.asarray(mesh.centroid, dtype=np.float64))

    return np.mean(np.stack(finger_centers, axis=0), axis=0)


def matrix_to_pose(matrix: np.ndarray) -> list[float]:
    """Converts a 4x4 transformation matrix to a pose (x, y, z, qx, qy, qz, qw).

    Args:
        matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        list[float]: Pose as [x, y, z, qx, qy, qz, qw].
    """
    x, y, z = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    quat = R.from_matrix(rot_mat).as_quat()
    qx, qy, qz, qw = quat

    return [x, y, z, qx, qy, qz, qw]


def pose_to_matrix(pose: list[float]) -> np.ndarray:
    """Converts pose (x, y, z, qx, qy, qz, qw) to a 4x4 transformation matrix.

    Args:
        pose (list[float]): Pose as [x, y, z, qx, qy, qz, qw].

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    x, y, z, qx, qy, qz, qw = pose
    r = R.from_quat([qx, qy, qz, qw])
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    matrix[:3, 3] = [x, y, z]

    return matrix


def compute_xy_bbox(
    vertices: np.ndarray, col_x: int = 0, col_y: int = 1
) -> list[float]:
    """Computes the bounding box in XY plane for given vertices.

    Args:
        vertices (np.ndarray): Vertex coordinates.
        col_x (int, optional): Column index for X.
        col_y (int, optional): Column index for Y.

    Returns:
        list[float]: [min_x, max_x, min_y, max_y]
    """
    x_vals = vertices[:, col_x]
    y_vals = vertices[:, col_y]
    return x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()


def has_iou_conflict(
    new_box: list[float],
    placed_boxes: list[list[float]],
    iou_threshold: float = 0.0,
) -> bool:
    """Checks for intersection-over-union conflict between boxes.

    Args:
        new_box (list[float]): New box coordinates.
        placed_boxes (list[list[float]]): List of placed box coordinates.
        iou_threshold (float, optional): IOU threshold.

    Returns:
        bool: True if conflict exists, False otherwise.
    """
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
    """Decorator to temporarily set the random seed for reproducibility.

    Args:
        seed_attr_name (str, optional): Name of the seed argument.

    Returns:
        function: Decorator function.
    """

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
    interp_per_edge: int = 10,
    margin: float = -0.02,
    x_axis: int = 0,
    y_axis: int = 1,
    z_axis: int = 2,
) -> Path:
    """Computes a dense convex hull path for the top surface of a mesh.

    Args:
        vertices (np.ndarray): Mesh vertices.
        z_threshold (float, optional): Z threshold for top surface.
        interp_per_edge (int, optional): Interpolation points per edge.
        margin (float, optional): Margin for polygon buffer.
        x_axis (int, optional): X axis index.
        y_axis (int, optional): Y axis index.
        z_axis (int, optional): Z axis index.

    Returns:
        Path: Matplotlib path object for the convex hull.
    """
    top_vertices = vertices[
        vertices[:, z_axis] > vertices[:, z_axis].max() - z_threshold
    ]
    top_xy = top_vertices[:, [x_axis, y_axis]]

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
    """Finds the parent node of a given node in a tree.

    Args:
        node (str): Node name.
        tree (dict): Tree structure.

    Returns:
        str | None: Parent node name or None.
    """
    for parent, children in tree.items():
        if any(child[0] == node for child in children):
            return parent
    return None


def all_corners_inside(hull: Path, box: list, threshold: int = 3) -> bool:
    """Checks if at least `threshold` corners of a box are inside a hull.

    Args:
        hull (Path): Convex hull path.
        box (list): Box coordinates [x1, x2, y1, y2].
        threshold (int, optional): Minimum corners inside.

    Returns:
        bool: True if enough corners are inside.
    """
    x1, x2, y1, y2 = box
    corners = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]

    num_inside = sum(hull.contains_point(c) for c in corners)
    return num_inside >= threshold


def compute_axis_rotation_quat(
    axis: Literal["x", "y", "z"], angle_rad: float
) -> list[float]:
    """Computes quaternion for rotation around a given axis.

    Args:
        axis (Literal["x", "y", "z"]): Axis of rotation.
        angle_rad (float): Rotation angle in radians.

    Returns:
        list[float]: Quaternion [x, y, z, w].
    """
    if axis.lower() == "x":
        q = Quaternion(axis=[1, 0, 0], angle=angle_rad)
    elif axis.lower() == "y":
        q = Quaternion(axis=[0, 1, 0], angle=angle_rad)
    elif axis.lower() == "z":
        q = Quaternion(axis=[0, 0, 1], angle=angle_rad)
    else:
        raise ValueError(f"Unsupported axis '{axis}', must be one of x, y, z")

    return [q.x, q.y, q.z, q.w]


def quaternion_multiply(
    init_quat: list[float], rotate_quat: list[float]
) -> list[float]:
    """Multiplies two quaternions.

    Args:
        init_quat (list[float]): Initial quaternion [x, y, z, w].
        rotate_quat (list[float]): Rotation quaternion [x, y, z, w].

    Returns:
        list[float]: Resulting quaternion [x, y, z, w].
    """
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
    """Checks if the target point is within the reachable range.

    Args:
        base_xyz (np.ndarray): Base position.
        reach_xyz (np.ndarray): Target position.
        min_reach (float, optional): Minimum reach distance.
        max_reach (float, optional): Maximum reach distance.

    Returns:
        bool: True if reachable, False otherwise.
    """
    distance = np.linalg.norm(reach_xyz - base_xyz)

    return min_reach < distance < max_reach


@with_seed("seed")
def bfs_placement(
    layout_file: str,
    floor_margin: float = 0,
    beside_margin: float = 0.1,
    max_attempts: int = 3000,
    init_rpy: tuple = (1.5708, 0.0, 0.0),
    rotate_objs: bool = True,
    rotate_bg: bool = True,
    rotate_context: bool = True,
    limit_reach_range: tuple[float, float] | None = (0.20, 0.85),
    max_orient_diff: float | None = 60,
    robot_dim: float = 0.12,
    seed: int = None,
) -> LayoutInfo:
    """Places objects in a scene layout using BFS traversal.

    Args:
        layout_file (str): Path to layout JSON file generated from `layout-cli`.
        floor_margin (float, optional): Z-offset for objects placed on the floor.
        beside_margin (float, optional): Minimum margin for objects placed 'beside' their parent, used when 'on' placement fails.
        max_attempts (int, optional): Max attempts for a non-overlapping placement.
        init_rpy (tuple, optional): Initial rotation (rpy).
        rotate_objs (bool, optional): Whether to random rotate objects.
        rotate_bg (bool, optional): Whether to random rotate background.
        rotate_context (bool, optional): Whether to random rotate context asset.
        limit_reach_range (tuple[float, float] | None, optional): If set, enforce a check that manipulated objects are within the robot's reach range, in meter.
        max_orient_diff (float | None, optional): If set, enforce a check that manipulated objects are within the robot's orientation range, in degree.
        robot_dim (float, optional): The approximate robot size.
        seed (int, optional): Random seed for reproducible placement.

    Returns:
        LayoutInfo: Layout information with object poses.

    Example:
        ```py
        from embodied_gen.utils.geometry import bfs_placement
        layout = bfs_placement("scene_layout.json", seed=42)
        print(layout.position)
        ```
    """
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
        rotation = R.from_euler("xyz", init_rpy, degrees=False)
        vertices = mesh.vertices @ rotation.as_matrix().T
        z1 = np.percentile(vertices[:, 2], 1)
        z2 = np.percentile(vertices[:, 2], 99)

        if object_mapping[node] == Scene3DItemEnum.CONTEXT.value:
            object_quat = [0, 0, 0, 1]
            if rotate_context:
                angle_rad = np.random.uniform(0, 2 * np.pi)
                object_quat = compute_axis_rotation_quat(
                    axis="z", angle_rad=angle_rad
                )
                rotation = R.from_quat(object_quat).as_matrix()
                vertices = vertices @ rotation.T

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
            vertices = vertices @ rotation.T

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
                        limit_reach_range is not None
                        and object_mapping[node]
                        == Scene3DItemEnum.MANIPULATED_OBJS.value
                    ):
                        cx = parent_pos[0] + node_box[0] + obj_dx / 2
                        cy = parent_pos[1] + node_box[2] + obj_dy / 2
                        cz = parent_pos[2] + p_z2 - z1
                        robot_pos = position[robot_node][:3]
                        if not check_reachable(
                            base_xyz=np.array(robot_pos),
                            reach_xyz=np.array([cx, cy, cz]),
                            min_reach=limit_reach_range[0],
                            max_reach=limit_reach_range[1],
                        ):
                            continue

                    # Make sure the manipulated object is inside the robot's orientation.
                    if (
                        max_orient_diff is not None
                        and object_mapping[node]
                        == Scene3DItemEnum.MANIPULATED_OBJS.value
                    ):
                        cx = parent_pos[0] + node_box[0] + obj_dx / 2
                        cy = parent_pos[1] + node_box[2] + obj_dy / 2
                        cx2, cy2 = position[robot_node][:2]
                        v1 = np.array([-cx2, -cy2])
                        v2 = np.array([cx - cx2, cy - cy2])
                        dot = np.dot(v1, v2)
                        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                        theta = np.arccos(np.clip(dot / norms, -1.0, 1.0))
                        theta = np.rad2deg(theta)
                        if theta > max_orient_diff:
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
    """Composes a mesh scene from layout information and saves to file.

    Args:
        layout_info (LayoutInfo): Layout information.
        out_scene_path (str): Output scene file path.
        with_bg (bool, optional): Include background mesh.
    """
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
    """Computes pinhole camera intrinsic matrix from image size and FOV.

    Args:
        image_w (int): Image width.
        image_h (int): Image height.
        fov_deg (float): Field of view in degrees.

    Returns:
        np.ndarray: Intrinsic matrix K.
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = image_w / (2 * np.tan(fov_rad / 2))
    fy = fx  # assuming square pixels
    cx = image_w / 2
    cy = image_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K
