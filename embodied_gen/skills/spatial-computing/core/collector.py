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

import hashlib
import logging
import os
import random
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import copy2, copytree
from typing import Any, Literal

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from shapely.affinity import translate
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

from .geometry import (
    DEFAULT_MESH_SAMPLE_NUM,
    get_actionable_surface,
    points_to_polygon,
)

logger = logging.getLogger(__name__)

# Type aliases
Geometry = Polygon | MultiPolygon

# Constants
DEFAULT_ROTATION_RPY = (1.57, 0.0, 0.0)
DEFAULT_MAX_PLACEMENT_ATTEMPTS = 3000
DEFAULT_IGNORE_ITEMS = ("ceiling", "light", "exterior")
DEFAULT_BESIDE_DISTANCE = 0.5
DEFAULT_Z_OFFSET = 0.001


def _exact_xy_projection(mesh: trimesh.Trimesh) -> Polygon:
    """Project all mesh triangles onto XY and union them into a footprint.

    Deterministic and shape-accurate (unlike point sampling), at the cost of
    iterating every triangle.

    Args:
        mesh: A world-transformed mesh.

    Returns:
        The projected footprint polygon, or an empty polygon.
    """
    triangle_polys = []
    for triangle in mesh.triangles[:, :, :2]:
        poly = Polygon(triangle)
        if poly.is_valid and poly.area > 1e-8:
            triangle_polys.append(poly)
    if not triangle_polys:
        return Polygon()
    return unary_union(triangle_polys).buffer(0)


def _load_mesh_to_poly(
    mesh_path: str,
    xyz: np.ndarray,
    rpy: np.ndarray,
    mesh_sample_num: int,
    use_exact_projection: bool = False,
) -> Polygon:
    """Load mesh and convert to 2D footprint polygon (process-safe).

    Standalone function for use with ProcessPoolExecutor.

    """
    if not os.path.exists(mesh_path):
        return Polygon()

    # Deterministic per-mesh seed so footprint sampling is reproducible
    # regardless of worker process or task ordering (str hashing is salted
    # across processes, so use a stable hash of the mesh path).
    mesh_seed = int.from_bytes(
        hashlib.md5(os.fspath(mesh_path).encode()).digest()[:4], "little"
    )
    np.random.seed(mesh_seed)

    mesh = trimesh.load(mesh_path, force="mesh", skip_materials=True)

    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    mesh.apply_transform(matrix)

    if use_exact_projection:
        projected_poly = _exact_xy_projection(mesh)
        if not projected_poly.is_empty:
            return projected_poly

    verts = np.asarray(mesh.sample(mesh_sample_num))[:, :2]
    poly = points_to_polygon(verts)

    # Scanline sampling can collapse for thin/rotated meshes (and is seed
    # sensitive); when the footprint is far smaller than the sampled-point
    # convex hull, fall back to the exact, deterministic projection.
    hull_area = MultiPoint(verts).convex_hull.area
    if hull_area > 1e-6 and poly.area < 0.5 * hull_area:
        projected_poly = _exact_xy_projection(mesh)
        if not projected_poly.is_empty and projected_poly.area > poly.area:
            return projected_poly

    return poly


class UrdfSemanticInfoCollector:
    """Collector for URDF semantic information.

    Parses URDF files to extract room layouts, object footprints, and
    provides methods for adding new instances and updating URDF/USD files.

    Attributes:
        mesh_sample_num: Number of points to sample from meshes.
        ignore_items: List of item name patterns to ignore.
        instances: Dictionary of instance name to footprint polygon.
        instance_meta: Dictionary of instance metadata (mesh path, pose).
        rooms: Dictionary of room polygons.
        footprints: Dictionary of object footprints.
        occ_area: Union of all occupied areas.
        floor_union: Union of all floor polygons.

    """

    def __init__(
        self,
        mesh_sample_num: int = DEFAULT_MESH_SAMPLE_NUM,
        ignore_items: list[str] | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            mesh_sample_num: Number of points to sample from meshes.
            ignore_items: List of item name patterns to ignore during parsing.

        """
        self.mesh_sample_num = mesh_sample_num
        self.ignore_items = ignore_items or list(DEFAULT_IGNORE_ITEMS)

        self.instances: dict[str, Polygon] = {}
        self.instance_meta: dict[str, dict] = {}
        self.rooms: dict[str, Geometry] = {}
        self.footprints: dict[str, Geometry] = {}
        self.occ_area: Geometry = Polygon()
        self.floor_union: Geometry = Polygon()

        self.urdf_path: str = ""
        self._tree: ET.ElementTree | None = None
        self._root: ET.Element | None = None

    def _get_transform(
        self,
        joint_elem: ET.Element,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract transform (xyz, rpy) from a joint element.

        Args:
            joint_elem: XML Element representing a URDF joint.

        Returns:
            Tuple of (xyz, rpy) arrays.

        """
        origin = joint_elem.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        else:
            xyz, rpy = np.zeros(3), np.zeros(3)
        return xyz, rpy

    def collect(self, urdf_path: str) -> None:
        """Parse URDF file and collect semantic information.

        Args:
            urdf_path: Path to the URDF file.

        """
        logger.info(f"Collecting URDF semantic info from {urdf_path}")
        self.urdf_path = urdf_path
        urdf_dir = os.path.dirname(urdf_path)

        self._tree = ET.parse(urdf_path)
        self._root = self._tree.getroot()

        link_transforms = self._build_link_transforms()
        self._process_links(urdf_dir, link_transforms)
        self._update_internal_state()

    def _build_link_transforms(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Build mapping from link names to their transforms.

        Returns:
            Dictionary mapping link names to (xyz, rpy) tuples.

        """
        link_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for joint in self._tree.findall("joint"):
            child = joint.find("child")
            if child is not None:
                link_name = child.attrib["link"]
                link_transforms[link_name] = self._get_transform(joint)

        return link_transforms

    def _process_links(
        self,
        urdf_dir: str,
        link_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Process all links in the URDF tree (parallel).

        Uses ProcessPoolExecutor to bypass GIL for CPU-bound mesh
        loading and sampling.

        Args:
            urdf_dir: Directory containing the URDF file.
            link_transforms: Dictionary of link transforms.

        """
        self.instances = {}
        self.instance_meta = {}
        wall_polys: list[Polygon] = []

        # Collect tasks for parallel processing
        tasks: list[dict] = []
        for link in self._tree.findall("link"):
            name = link.attrib.get("name", "").lower()
            if any(ign in name for ign in self.ignore_items):
                continue

            visual = link.find("visual")
            if visual is None:
                continue

            mesh_node = visual.find("geometry/mesh")
            if mesh_node is None:
                continue

            mesh_path = os.path.join(urdf_dir, mesh_node.attrib["filename"])
            default_transform = (np.zeros(3), np.zeros(3))
            xyz, rpy = link_transforms.get(
                link.attrib["name"], default_transform
            )
            tasks.append(
                {
                    "link_name": link.attrib["name"],
                    "link_name_lower": name,
                    "mesh_path": mesh_path,
                    "xyz": xyz,
                    "rpy": rpy,
                }
            )

        logger.info(
            "Processing %d URDF links to extract geometry "
            "(parallel, sample_num=%d)...",
            len(tasks),
            self.mesh_sample_num,
        )

        # ProcessPoolExecutor bypasses GIL for CPU-bound trimesh ops.
        # Cap workers to balance parallelism vs memory overhead.
        n_workers = min(len(tasks), os.cpu_count() or 4, 8)
        futures_map: dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for task in tasks:
                future = executor.submit(
                    _load_mesh_to_poly,
                    task["mesh_path"],
                    task["xyz"],
                    task["rpy"],
                    self.mesh_sample_num,
                    "_floor" in task["link_name_lower"],
                )
                futures_map[future] = task

            for future in as_completed(futures_map):
                task = futures_map[future]
                try:
                    poly = future.result()
                except Exception:
                    logger.warning(
                        "Failed to process link '%s', skipping.",
                        task["link_name"],
                        exc_info=True,
                    )
                    continue

                if poly.is_empty:
                    continue

                if "wall" in task["link_name_lower"]:
                    wall_polys.append(poly)
                else:
                    key = self._process_safe_key_robust(task["link_name"])
                    self.instances[key] = poly
                    self.instance_meta[key] = {
                        "mesh_path": task["mesh_path"],
                        "xyz": task["xyz"],
                        "rpy": task["rpy"],
                        "original_link_name": task["link_name"],
                    }

        self.instances["walls"] = unary_union(wall_polys)

    def _update_internal_state(self) -> None:
        """Update derived state (rooms, footprints, occupied area)."""
        self.rooms = {
            k: v
            for k, v in self.instances.items()
            if "_floor" in k.lower() and not v.is_empty
        }

        self.footprints = {
            k: v
            for k, v in self.instances.items()
            if k != "walls"
            and "_floor" not in k.lower()
            and "rug" not in k.lower()
            and not v.is_empty
        }
        self.occ_area = unary_union(list(self.footprints.values()))
        self.floor_union = unary_union(list(self.rooms.values()))

    def _process_safe_key_robust(self, name: str) -> str:
        """Convert a link name to a safe, normalized key.

        Args:
            name: Original link name.

        Returns:
            Normalized key string.

        """
        if name.endswith("_floor"):
            parts = name.split("_")
            return "_".join(parts[:-2] + ["floor"])

        if "Factory" in name:
            # Handle infinigen naming convention
            prefix = name.split("Factory")[0]
            suffix = f"_{name.split('_')[-1]}"
        else:
            prefix, suffix = name, ""

        res = prefix.replace(" ", "_")
        res = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", res)
        res = res.lower()
        res = re.sub(r"_+", "_", res).strip("_ ")

        return f"{res}{suffix}"

    def add_instance(
        self,
        asset_path: str,
        instance_key: str,
        in_room: str | None = None,
        on_instance: str | None = None,
        beside_instance: str | None = None,
        beside_distance: float = DEFAULT_BESIDE_DISTANCE,
        rotation_rpy: tuple[float, float, float] = DEFAULT_ROTATION_RPY,
        n_max_attempt: int = DEFAULT_MAX_PLACEMENT_ATTEMPTS,
        place_strategy: Literal["top", "random"] = "random",
    ) -> list[float] | None:
        """Add a new instance to the scene with automatic placement.

        Args:
            asset_path: Path to the asset mesh file.
            instance_key: Unique key for the new instance.
            in_room: Optional room name to constrain placement.
            on_instance: Optional instance name to place on top of.
            beside_instance: Optional instance name to place beside (on floor).
            beside_distance: Initial buffer distance from the target instance
                for beside placement (meters). Will auto-expand if needed.
            rotation_rpy: Initial rotation in roll-pitch-yaw.
            n_max_attempt: Maximum placement attempts.
            place_strategy: Either "top" or "random".

        Returns:
            List [x, y, z] of the placed instance center, or None if failed.

        Raises:
            ValueError: If instance_key already exists or room/instance not found.

        """
        if instance_key in self.instances:
            raise ValueError(f"Instance key '{instance_key}' already exists.")

        room_poly = self._resolve_room_polygon(in_room)

        # Load mesh and compute base polygon (needed for all placement modes)
        mesh = trimesh.load(asset_path, force="mesh")
        mesh.apply_transform(
            trimesh.transformations.euler_matrix(*rotation_rpy, "sxyz")
        )

        verts = np.asarray(mesh.sample(self.mesh_sample_num))[:, :2]
        base_poly = points_to_polygon(verts)
        centroid = base_poly.centroid
        base_poly = translate(base_poly, xoff=-centroid.x, yoff=-centroid.y)

        if beside_instance is not None:
            placement = self._try_place_beside(
                base_poly=base_poly,
                beside_instance=beside_instance,
                room_poly=room_poly,
                beside_distance=beside_distance,
                n_max_attempt=n_max_attempt,
                multi_match_strategy="first",  # Default strategy
            )
            base_z = 0.0
        else:
            target_area, obstacles, base_z = self._resolve_placement_target(
                on_instance, room_poly, place_strategy
            )

            if target_area.is_empty:
                logger.error("Target area for placement is empty.")
                return None

            placement = self._try_place_polygon(
                base_poly, target_area, obstacles, n_max_attempt
            )

        if placement is None:
            logger.error(
                f"Failed to place '{instance_key}' after all attempts."
            )
            return None

        x, y, candidate = placement
        self.instances[instance_key] = candidate
        final_z = base_z - mesh.bounds[0][2] + DEFAULT_Z_OFFSET
        self._update_internal_state()

        return [round(v, 4) for v in (x, y, final_z)]

    def _resolve_room_polygon(self, in_room: str | None) -> Geometry | None:
        """Resolve room name to polygon.

        Args:
            in_room: Room name query string.

        Returns:
            Room polygon or None if not specified.

        Raises:
            ValueError: If room not found.

        """
        if in_room is None:
            return None

        query_room = in_room.lower()
        room_matches = [
            k for k in self.rooms.keys() if query_room in k.lower()
        ]

        if not room_matches:
            raise ValueError(f"Room '{in_room}' not found.")

        return unary_union([self.rooms[k] for k in room_matches])

    def _try_place_beside(
        self,
        base_poly: Polygon,
        beside_instance: str,
        room_poly: Geometry | None,
        beside_distance: float = DEFAULT_BESIDE_DISTANCE,
        n_max_attempt: int = DEFAULT_MAX_PLACEMENT_ATTEMPTS,
        max_expand_steps: int = 5,
        expand_factor: float = 1.5,
        multi_match_strategy: Literal["first", "random", "largest"] = "first",
    ) -> tuple[float, float, Polygon] | None:
        """Place object beside target with progressive distance expansion.

        More robust than fixed-distance placement:
        1. Ensures minimum distance accommodates the new object's size.
        2. Pre-subtracts obstacles from the ring → sampling only in free area.
        3. Progressively expands distance on failure (up to max_expand_steps).
        4. Skips steps where the free area is too small for the object.

        Args:
            base_poly: Object footprint polygon centered at origin.
            beside_instance: Target instance name to place beside.
            room_poly: Optional room constraint polygon.
            beside_distance: Initial buffer distance (meters).
            n_max_attempt: Total max placement attempts across all steps.
            max_expand_steps: Max number of distance expansion rounds.
            expand_factor: Distance multiplier per expansion round.
            multi_match_strategy: How to pick among multiple matching target
                instances ("first", "random", or "largest").

        Returns:
            Tuple (x, y, placed_polygon) on success, or None if all failed.

        Raises:
            ValueError: If beside_instance not found in scene.

        """
        # --- Resolve target instance ---
        query_obj = beside_instance.lower()
        possible_matches = [
            k
            for k in self.instances.keys()
            if query_obj in k.lower() and k != "walls"
        ]

        if room_poly is not None:
            # Check that the object's representative point falls inside
            # the room (buffered slightly for mesh-sampling tolerance).
            room_buffered = room_poly.buffer(0.1)
            possible_matches = [
                k
                for k in possible_matches
                if room_buffered.contains(
                    self.instances[k].representative_point()
                )
            ]

        if not possible_matches:
            location_msg = " in specified room" if room_poly else ""
            # Log candidate distances for easier debugging
            all_matches = [
                k
                for k in self.instances.keys()
                if query_obj in k.lower() and k != "walls"
            ]
            if all_matches and room_poly is not None:
                dists = {
                    k: round(self.instances[k].distance(room_poly), 4)
                    for k in all_matches
                }
                logger.error("Candidate distances to room polygon: %s", dists)
            raise ValueError(
                f"No instance matching '{beside_instance}' "
                f"found{location_msg}."
            )

        if len(possible_matches) > 1:
            # Apply multi-match strategy
            if multi_match_strategy == "random":
                target_key = random.choice(possible_matches)
            elif multi_match_strategy == "largest":
                target_key = max(
                    possible_matches, key=lambda k: self.instances[k].area
                )
            else:  # "first"
                target_key = possible_matches[0]
            logger.warning(
                f"Multiple matches for '{beside_instance}': "
                f"{possible_matches}. Using '{target_key}' "
                f"(strategy: {multi_match_strategy})."
            )
        else:
            target_key = possible_matches[0]

        target_footprint = self.instances[target_key]
        floor = room_poly if room_poly is not None else self.floor_union

        # --- Ensure initial distance accommodates the object's size ---
        obj_bounds = base_poly.bounds  # (minx, miny, maxx, maxy)
        obj_half_diag = (
            np.hypot(
                obj_bounds[2] - obj_bounds[0],
                obj_bounds[3] - obj_bounds[1],
            )
            / 2.0
        )
        current_distance = max(beside_distance, obj_half_diag * 1.5)

        # Budget attempts across expansion steps
        attempts_per_step = max(n_max_attempt // (max_expand_steps + 1), 500)
        empty_obstacle = Polygon()  # pre-created; obstacles are pre-subtracted

        for step in range(max_expand_steps + 1):
            # Build ring: buffer - footprint, intersected with floor
            buffered = target_footprint.buffer(current_distance)
            ring_area = buffered.difference(target_footprint)
            ring_area = ring_area.intersection(floor)

            if ring_area.is_empty:
                logger.info(
                    f"[beside step {step}] Ring empty at "
                    f"{current_distance:.2f}m, expanding..."
                )
                current_distance *= expand_factor
                continue

            # Pre-subtract all obstacles → sample only from actual free area
            free_area = ring_area.difference(self.occ_area)

            if free_area.is_empty or free_area.area < base_poly.area * 0.5:
                logger.info(
                    f"[beside step {step}] Free area too small at "
                    f"{current_distance:.2f}m "
                    f"(free={free_area.area:.4f}, "
                    f"need≈{base_poly.area:.4f}), expanding..."
                )
                current_distance *= expand_factor
                continue

            # Attempt placement in the free area (obstacles already removed)
            placement = self._try_place_polygon(
                base_poly, free_area, empty_obstacle, attempts_per_step
            )

            if placement is not None:
                logger.info(
                    f"Placed beside '{target_key}' at distance "
                    f"{current_distance:.2f}m (step {step})"
                )
                return placement

            logger.info(
                f"[beside step {step}] Failed at {current_distance:.2f}m "
                f"after {attempts_per_step} attempts, expanding..."
            )
            current_distance *= expand_factor

        logger.error(
            f"Failed to place beside '{target_key}' after "
            f"{max_expand_steps + 1} expansion steps "
            f"(final distance: {current_distance / expand_factor:.2f}m)."
        )
        return None

    def _resolve_placement_target(
        self,
        on_instance: str | None,
        room_poly: Geometry | None,
        place_strategy: Literal["top", "random"],
    ) -> tuple[Geometry, Geometry, float]:
        """Resolve the target placement area and obstacles.

        Args:
            on_instance: Instance name to place on.
            room_poly: Room polygon constraint.
            place_strategy: Placement strategy.

        Returns:
            Tuple of (target_area, obstacles, base_z_height).

        Raises:
            ValueError: If on_instance not found.

        """
        if on_instance is None:
            if room_poly is not None:
                return room_poly, self.occ_area, 0.0
            return self.floor_union, self.occ_area, 0.0

        query_obj = on_instance.lower()
        possible_matches = [
            k
            for k in self.instances.keys()
            if query_obj in k.lower() and k != "walls"
        ]

        if room_poly is not None:
            room_buffered = room_poly.buffer(0.1)
            possible_matches = [
                k
                for k in possible_matches
                if room_buffered.contains(
                    self.instances[k].representative_point()
                )
            ]

        if not possible_matches:
            location_msg = f" in room '{on_instance}'" if room_poly else ""
            raise ValueError(
                f"No instance matching '{on_instance}' found{location_msg}."
            )

        if place_strategy == "random":
            target_parent_key = random.choice(possible_matches)
        else:
            target_parent_key = possible_matches[0]

        if len(possible_matches) > 1:
            logger.warning(
                f"Multiple matches for '{on_instance}': {possible_matches}. "
                f"Using '{target_parent_key}'."
            )

        meta = self.instance_meta[target_parent_key]
        parent_mesh = trimesh.load(meta["mesh_path"], force="mesh")
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_euler("xyz", meta["rpy"]).as_matrix()
        matrix[:3, 3] = meta["xyz"]
        parent_mesh.apply_transform(matrix)

        best_z, surface_poly = get_actionable_surface(
            parent_mesh, place_strategy=place_strategy
        )
        obstacles = self.occ_area.difference(self.instances[target_parent_key])

        # Re-add footprints of objects inside the parent polygon so they
        # remain obstacles (difference above removes them).
        parent_poly = self.instances[target_parent_key]
        children_on_parent = [
            poly
            for key, poly in self.footprints.items()
            if key != target_parent_key and parent_poly.contains(poly)
        ]
        if children_on_parent:
            obstacles = unary_union([obstacles] + children_on_parent)

        logger.info(f"Placing on '{target_parent_key}' (Z={best_z:.3f})")

        return surface_poly, obstacles, best_z

    def _try_place_polygon(
        self,
        base_poly: Polygon,
        target_area: Geometry,
        obstacles: Geometry,
        n_max_attempt: int,
    ) -> tuple[float, float, Polygon] | None:
        """Try to place polygon in target area avoiding obstacles.

        Pre-computes the free area (target minus obstacles) so that the
        containment check alone is sufficient, avoiding redundant
        intersection tests against obstacles on every iteration.

        Args:
            base_poly: Polygon to place (centered at origin).
            target_area: Area where placement is allowed.
            obstacles: Areas to avoid.
            n_max_attempt: Maximum attempts.

        Returns:
            Tuple of (x, y, placed_polygon) or None if failed.

        """
        if not obstacles.is_empty:
            free_area = target_area.difference(obstacles)
        else:
            free_area = target_area

        if free_area.is_empty:
            return None

        minx, miny, maxx, maxy = free_area.bounds

        for _ in range(n_max_attempt):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            candidate = translate(base_poly, xoff=x, yoff=y)

            if free_area.contains(candidate):
                return x, y, candidate

        return None

    def update_urdf_info(
        self,
        output_path: str,
        instance_key: str,
        visual_mesh_path: str,
        collision_mesh_path: str | None = None,
        trans_xyz: tuple[float, float, float] = (0, 0, 0),
        rot_rpy: tuple[float, float, float] = DEFAULT_ROTATION_RPY,
        joint_type: str = "fixed",
    ) -> None:
        """Add a new link to the URDF tree and save.

        Args:
            output_path: Path to save the updated URDF.
            instance_key: Name for the new link.
            visual_mesh_path: Path to the visual mesh file.
            collision_mesh_path: Optional path to collision mesh.
            trans_xyz: Translation (x, y, z).
            rot_rpy: Rotation (roll, pitch, yaw).
            joint_type: Type of joint (e.g., "fixed").

        """
        if self._root is None:
            return

        logger.info(f"Updating URDF for instance '{instance_key}'.")
        urdf_dir = os.path.dirname(self.urdf_path)

        # Copy mesh files
        copytree(
            os.path.dirname(visual_mesh_path),
            f"{urdf_dir}/{instance_key}",
            dirs_exist_ok=True,
        )
        visual_rel_path = (
            f"{instance_key}/{os.path.basename(visual_mesh_path)}"
        )

        collision_rel_path = None
        if collision_mesh_path is not None:
            copytree(
                os.path.dirname(collision_mesh_path),
                f"{urdf_dir}/{instance_key}",
                dirs_exist_ok=True,
            )
            collision_rel_path = (
                f"{instance_key}/{os.path.basename(collision_mesh_path)}"
            )

        # Create link element
        link = ET.SubElement(self._root, "link", attrib={"name": instance_key})

        visual = ET.SubElement(link, "visual")
        v_geo = ET.SubElement(visual, "geometry")
        ET.SubElement(v_geo, "mesh", attrib={"filename": visual_rel_path})

        if collision_rel_path is not None:
            collision = ET.SubElement(link, "collision")
            c_geo = ET.SubElement(collision, "geometry")
            ET.SubElement(
                c_geo, "mesh", attrib={"filename": collision_rel_path}
            )

        # Create joint element
        joint_name = f"joint_{instance_key}"
        joint = ET.SubElement(
            self._root,
            "joint",
            attrib={"name": joint_name, "type": joint_type},
        )

        ET.SubElement(joint, "parent", attrib={"link": "base"})
        ET.SubElement(joint, "child", attrib={"link": instance_key})

        xyz_str = f"{trans_xyz[0]:.4f} {trans_xyz[1]:.4f} {trans_xyz[2]:.4f}"
        rpy_str = f"{rot_rpy[0]:.4f} {rot_rpy[1]:.4f} {rot_rpy[2]:.4f}"
        ET.SubElement(joint, "origin", attrib={"xyz": xyz_str, "rpy": rpy_str})

        self.save_urdf(output_path)

    def update_usd_info(
        self,
        usd_path: str,
        output_path: str,
        instance_key: str,
        visual_mesh_path: str,
        trans_xyz: list[float],
        rot_rpy: tuple[float, float, float] = DEFAULT_ROTATION_RPY,
    ) -> None:
        """Add a mesh instance to an existing USD file.

        Uses Blender (bpy) to convert OBJ to USD format.

        Args:
            usd_path: Path to the source USD file.
            output_path: Path to save the modified USD.
            instance_key: Prim path name for the new instance.
            visual_mesh_path: Path to the visual mesh (OBJ format).
            trans_xyz: Translation [x, y, z].
            rot_rpy: Rotation (roll, pitch, yaw).

        Raises:
            ImportError: If pxr (USD) library or bpy is not available.

        """
        import bpy
        from pxr import Gf, Usd, UsdGeom

        out_dir = os.path.dirname(output_path)
        target_dir = os.path.join(out_dir, instance_key)
        os.makedirs(target_dir, exist_ok=True)

        mesh_filename = os.path.basename(visual_mesh_path)
        usdc_filename = os.path.splitext(mesh_filename)[0] + ".usdc"
        target_usdc_path = os.path.join(target_dir, usdc_filename)

        logger.info(
            f"Converting with Blender (bpy): "
            f"{visual_mesh_path} -> {target_usdc_path}"
        )
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.obj_import(
            filepath=visual_mesh_path,
            forward_axis="Y",
            up_axis="Z",
        )
        bpy.ops.wm.usd_export(
            filepath=target_usdc_path,
            selected_objects_only=False,
        )

        # Copy texture files
        src_dir = os.path.dirname(visual_mesh_path)
        for f in os.listdir(src_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".mtl")):
                copy2(os.path.join(src_dir, f), target_dir)

        final_rel_path = f"./{instance_key}/{usdc_filename}"

        # Update USD stage
        stage = Usd.Stage.Open(usd_path)
        prim_path = self._usd_instance_prim_path(stage, instance_key)
        mesh_prim = UsdGeom.Xform.Define(stage, prim_path)

        ref_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
        ref_prim.GetPrim().GetReferences().AddReference(final_rel_path)

        # Build transform matrix
        translation_mat = Gf.Matrix4d().SetTranslate(
            Gf.Vec3d(trans_xyz[0], trans_xyz[1], trans_xyz[2])
        )
        rx = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), np.degrees(rot_rpy[0]))
        )
        ry = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(0, 1, 0), np.degrees(rot_rpy[1]))
        )
        rz = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), np.degrees(rot_rpy[2]))
        )
        rotation_mat = rx * ry * rz
        transform = rotation_mat * translation_mat
        mesh_prim.AddTransformOp().Set(transform)

        stage.GetRootLayer().Export(output_path)
        logger.info(f"✅ Saved updated USD to {output_path}")

    @staticmethod
    def _usd_scene_root_path(stage: Any) -> str:
        """Return the prim path that should own scene-authored assets."""
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            return default_prim.GetPath().pathString

        world_prim = stage.GetPrimAtPath("/World")
        if world_prim and world_prim.IsValid():
            stage.SetDefaultPrim(world_prim)
            logger.info(
                "USD has no defaultPrim; using '/World' as scene root."
            )
            return "/World"

        return ""

    @classmethod
    def _usd_instance_prim_path(cls, stage: Any, instance_key: str) -> str:
        """Build a USD prim path under the stage defaultPrim when available."""
        scene_root_path = cls._usd_scene_root_path(stage)
        if scene_root_path:
            return f"{scene_root_path}/{instance_key}"
        return f"/{instance_key}"

    def remove_usd_instance(
        self,
        usd_path: str,
        output_path: str,
        instance_key: str,
    ) -> None:
        """Remove an instance from a USD file.

        Args:
            usd_path: Path to the source USD file.
            output_path: Path to save the modified USD.
            instance_key: Prim path name of the instance to remove.

        Raises:
            ImportError: If pxr (USD) library is not available.

        """
        from pxr import Usd

        # Open USD stage
        stage = Usd.Stage.Open(usd_path)

        # Find and remove the prim. Check the defaultPrim path first, and
        # keep the old root-level path as a compatibility fallback.
        prim_paths = [self._usd_instance_prim_path(stage, instance_key)]
        legacy_prim_path = f"/{instance_key}"
        if legacy_prim_path not in prim_paths:
            prim_paths.append(legacy_prim_path)

        removed = False
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.RemovePrim(prim_path)
                logger.info(f"Removed prim '{prim_path}' from USD.")
                removed = True

        if not removed:
            logger.warning(
                f"Prim '{instance_key}' not found in USD stage under "
                "defaultPrim or legacy root path."
            )

        # Export modified stage
        stage.GetRootLayer().Export(output_path)
        logger.info(f"✅ Saved updated USD to {output_path}")

    def remove_instance(
        self,
        instance_key: str,
        in_room: str | None = None,
    ) -> bool:
        """Remove an instance from the scene.

        Args:
            instance_key: Exact instance name or semantic description to remove.
            in_room: Optional room constraint - only remove if instance is in this room.

        Returns:
            True if instance was removed, False if not found.

        Raises:
            ValueError: If instance_key is a protected item (walls, floors).

        """
        # Protect critical items
        protected = ["walls"] + [
            k for k in self.instances.keys() if "floor" in k.lower()
        ]
        if instance_key in protected:
            raise ValueError(
                f"Cannot remove protected instance '{instance_key}'. "
                f"Protected items: {protected}"
            )

        # Check if instance exists
        if instance_key not in self.instances:
            logger.warning(f"Instance '{instance_key}' not found in scene.")
            return False

        # Check room constraint if specified
        if in_room is not None:
            room_poly = self._resolve_room_polygon(in_room)
            if room_poly is not None:
                room_buffered = room_poly.buffer(0.1)
                instance_point = self.instances[
                    instance_key
                ].representative_point()
                if not room_buffered.contains(instance_point):
                    logger.warning(
                        f"Instance '{instance_key}' is not in room '{in_room}'."
                    )
                    return False

        # Remove from URDF XML tree
        if self._root is not None:
            self._remove_link_and_joint(instance_key)

        # Remove from instances dict
        del self.instances[instance_key]

        # Remove from metadata
        if instance_key in self.instance_meta:
            del self.instance_meta[instance_key]

        # Update internal state
        self._update_internal_state()

        logger.info(f"✅ Removed instance '{instance_key}' from scene.")
        return True

    def _remove_link_and_joint(self, instance_key: str) -> None:
        """Remove link and joint elements from URDF XML tree.

        Args:
            instance_key: Key of the instance to remove (simplified key).

        """
        if self._root is None:
            return

        # Get original link name from metadata
        meta = self.instance_meta.get(instance_key, {})
        original_link_name = meta.get("original_link_name", instance_key)

        # Find and remove the link element
        link_removed = False
        for link in self._root.findall("link"):
            if link.attrib.get("name") == original_link_name:
                self._root.remove(link)
                logger.info(f"Removed link '{original_link_name}' from URDF.")
                link_removed = True
                break

        if not link_removed:
            logger.warning(
                f"Link '{original_link_name}' not found in URDF tree."
            )

        # Find and remove the joint element
        joint_removed = False
        for joint in self._root.findall("joint"):
            child = joint.find("child")
            if (
                child is not None
                and child.attrib.get("link") == original_link_name
            ):
                self._root.remove(joint)
                logger.info(
                    f"Removed joint for '{original_link_name}' from URDF."
                )
                joint_removed = True
                break

        if not joint_removed:
            logger.warning(
                f"Joint for '{original_link_name}' not found in URDF tree."
            )

    def get_instance_center(self, instance_key: str) -> list[float] | None:
        """Get the center position of an instance.

        Args:
            instance_key: Name of the instance to query.

        Returns:
            List [x, y, z] of the instance center, or None if not found.

        """
        if instance_key not in self.instances:
            logger.warning(f"Instance '{instance_key}' not found in scene.")
            return None

        # Get instance metadata
        meta = self.instance_meta.get(instance_key, {})
        xyz = meta.get("xyz", np.zeros(3))

        # Get polygon centroid for 2D position
        poly = self.instances[instance_key]
        centroid = poly.centroid

        # Return [x, y, z] where x,y are from polygon centroid, z from metadata
        center = [round(centroid.x, 4), round(centroid.y, 4), round(xyz[2], 4)]

        logger.info(f"Instance '{instance_key}' center: {center}")
        return center

    def save_urdf(self, output_path: str) -> None:
        """Save the current URDF tree to file.

        Args:
            output_path: Path to save the URDF file.

        """
        if self._tree is None:
            return

        if hasattr(ET, "indent"):
            ET.indent(self._tree, space="  ", level=0)

        self._tree.write(output_path, encoding="utf-8", xml_declaration=True)
        logger.info(f"✅ Saved updated URDF to {output_path}")
