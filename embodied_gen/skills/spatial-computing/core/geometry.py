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

import logging
import random
from typing import Literal

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, MultiPolygon, Polygon

logger = logging.getLogger(__name__)

# Type aliases
Geometry = Polygon | MultiPolygon

# Constants
DEFAULT_MESH_SAMPLE_NUM = 10000
DEFAULT_MAX_PLACEMENT_ATTEMPTS = 2000


def points_to_polygon(
    points: np.ndarray,
    smooth_thresh: float = 0.2,
    scanline_step: float = 0.01,
) -> Polygon:
    """Convert point clouds into polygon contours using sweep line algorithm.

    Args:
        points: Array of 2D points with shape (N, 2).
        smooth_thresh: Buffer threshold for smoothing the polygon.
        scanline_step: Step size for the scanline sweep.

    Returns:
        A Shapely Polygon representing the contour of the point cloud.

    """
    if len(points) == 0:
        return Polygon()

    ys = points[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_values = np.arange(y_min, y_max + scanline_step, scanline_step)

    upper: list[list[float]] = []
    lower: list[list[float]] = []

    for y in y_values:
        pts_in_strip = points[(ys >= y) & (ys < y + scanline_step)]
        if len(pts_in_strip) == 0:
            continue

        xs = pts_in_strip[:, 0]
        upper.append([xs.max(), y])
        lower.append([xs.min(), y])

    contour = upper + lower[::-1]
    if len(contour) < 3:
        return Polygon()

    poly = Polygon(contour)
    return poly.buffer(smooth_thresh).buffer(-smooth_thresh)


def get_actionable_surface(
    mesh: trimesh.Trimesh,
    tol_angle: int = 10,
    tol_z: float = 0.02,
    area_tolerance: float = 0.15,
    place_strategy: Literal["top", "random"] = "random",
) -> tuple[float, Geometry]:
    """Extract the actionable (placeable) surface from a mesh.

    Finds upward-facing surfaces and returns the best one based on the
    placement strategy.

    Args:
        mesh: The input trimesh object.
        tol_angle: Angle tolerance in degrees for detecting up-facing normals.
        tol_z: Z-coordinate tolerance for clustering faces.
        area_tolerance: Tolerance for selecting candidate surfaces by area.
        place_strategy: Either "top" (highest surface) or "random".

    Returns:
        A tuple of (z_height, surface_polygon) representing the selected
        actionable surface.

    """
    up_vec = np.array([0, 0, 1])
    dots = np.dot(mesh.face_normals, up_vec)
    valid_mask = dots > np.cos(np.deg2rad(tol_angle))

    if not np.any(valid_mask):
        logger.warning(
            "No up-facing surfaces found. Falling back to bounding box top."
        )
        verts = mesh.vertices[:, :2]
        return mesh.bounds[1][2], MultiPoint(verts).convex_hull

    valid_faces_indices = np.where(valid_mask)[0]
    face_z = mesh.triangles_center[valid_mask][:, 2]
    face_areas = mesh.area_faces[valid_mask]

    z_clusters = _cluster_faces_by_z(
        face_z, face_areas, valid_faces_indices, tol_z
    )

    if not z_clusters:
        return mesh.bounds[1][2], MultiPoint(mesh.vertices[:, :2]).convex_hull

    selected_z, selected_data = _select_surface_cluster(
        z_clusters, area_tolerance, place_strategy
    )

    # For "top" strategy, use the highest z among all clusters for
    # base height, while keeping the largest-area polygon for XY placement.
    if place_strategy == "top":
        highest_z = max(z_clusters.keys())
        if highest_z > selected_z:
            logger.info(
                f"Overriding base Z from {selected_z:.3f} to "
                f"highest surface {highest_z:.3f}"
            )
            selected_z = highest_z

    cluster_faces = mesh.faces[selected_data["indices"]]
    temp_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=cluster_faces)
    samples, _ = trimesh.sample.sample_surface(temp_mesh, 10000)

    if len(samples) < 3:
        logger.warning(
            f"Failed to sample enough points on layer Z={selected_z}. "
            "Returning empty polygon."
        )
        return selected_z, Polygon()

    surface_poly = MultiPoint(samples[:, :2]).convex_hull
    return selected_z, surface_poly


def _cluster_faces_by_z(
    face_z: np.ndarray,
    face_areas: np.ndarray,
    face_indices: np.ndarray,
    tol_z: float,
) -> dict[float, dict]:
    """Cluster mesh faces by their Z coordinate.

    Args:
        face_z: Z coordinates of face centers.
        face_areas: Areas of each face.
        face_indices: Original indices of the faces.
        tol_z: Tolerance for Z clustering.

    Returns:
        Dictionary mapping Z values to cluster data (area and indices).

    """
    z_clusters: dict[float, dict] = {}

    for i, z in enumerate(face_z):
        key = round(z / tol_z) * tol_z

        if key not in z_clusters:
            z_clusters[key] = {"area": 0.0, "indices": []}

        z_clusters[key]["area"] += face_areas[i]
        z_clusters[key]["indices"].append(face_indices[i])

    return z_clusters


def _select_surface_cluster(
    z_clusters: dict[float, dict],
    area_tolerance: float,
    place_strategy: Literal["top", "random"],
) -> tuple[float, dict]:
    """Select the best surface cluster based on strategy.

    Args:
        z_clusters: Dictionary of Z clusters with area and indices.
        area_tolerance: Tolerance for candidate selection by area.
        place_strategy: Either "top" or "random".

    Returns:
        Tuple of (selected_z, cluster_data).

    """
    max_area = max(c["area"] for c in z_clusters.values())
    candidates = [
        (z, data)
        for z, data in z_clusters.items()
        if data["area"] >= max_area * (1.0 - area_tolerance)
    ]

    if not candidates:
        best_item = max(z_clusters.items(), key=lambda x: x[1]["area"])
        candidates = [best_item]

    if place_strategy == "random":
        selected_z, selected_data = random.choice(candidates)
        logger.info(
            f"Strategy 'random': Selected Z={selected_z:.3f} "
            f"(Area={selected_data['area']:.3f}) "
            f"from {len(candidates)} candidates."
        )
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_z, selected_data = candidates[0]
        logger.info(
            f"Strategy 'top': Selected highest Z={selected_z:.3f} "
            f"(Area={selected_data['area']:.3f})"
        )

    return selected_z, selected_data
