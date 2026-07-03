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

import heapq
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Type aliases
Geometry = Polygon | MultiPolygon

# Constants
DEFAULT_CLEARANCE = 0.4
"""Minimum distance (m) required between the trajectory and any obstacle."""

DEFAULT_RESOLUTION = 0.05
"""Occupancy-grid cell size in meters."""

DEFAULT_OBSTACLE_IGNORE = ("door",)
"""Footprint name keywords excluded from obstacles (treated as open)."""

DEFAULT_NUM_WAYPOINTS = 8
"""Number of roam waypoints sampled across the navigable space."""

DEFAULT_POINT_SPACING = 0.1
"""Output spacing (m) between consecutive trajectory points."""

DEFAULT_TURN_RADIUS = 0.5
"""Target turning-arc radius (m) for rounding corners."""

DEFAULT_ENDPOINT_CLEARANCE = 1.5
"""Minimum distance (m) the start/end keep from walls and objects."""

HAIRPIN_REVERSAL_DOT = -0.85
"""In/out direction dot product below which a turn is treated as a hairpin
(near 180 deg) and collapsed; normal roam turns are preserved."""

TURN_FRAME_TRIGGER_DEG = 45.0
"""Only heading jumps larger than this (a big U-turn at an unavoidable sharp
corner) get extra in-place rotation frames; gentler turns flow continuously."""

TURN_FRAME_STEP_DEG = 20.0
"""Heading step (deg) of each inserted in-place rotation frame."""

MIN_FILLET_RADIUS = 0.15
"""Smallest fillet radius (m) tried when rounding a corner; below this a
corner that still cannot be rounded is left sharp."""

# 8-connected neighbor offsets (di, dj) with step cost.
_NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
]


@dataclass
class TrajectoryResult:
    """Result of a roaming-trajectory generation.

    Attributes:
        points: Array (N, 3) of (x, y, rot_deg). rot_deg is the heading
            tangent to the curve: 0 deg points +Y (12 o'clock), increasing
            counter-clockwise in [0, 360).
        clearance: Clearance radius (m) used for planning.
        min_clearance: Minimum obstacle clearance (m) along the trajectory.
        length: Total arc length (m) of the trajectory.
        reachable_rooms: Room keys reachable within the planning component.
    """

    points: np.ndarray
    clearance: float
    min_clearance: float
    length: float
    reachable_rooms: list[str] = field(default_factory=list)


def heading_to_rot_deg(dx: float, dy: float) -> float:
    """Convert a motion direction to the roaming heading convention.

    The heading is the forward (tangent) direction of travel, where 0 deg
    points to +Y (12 o'clock) and the angle increases counter-clockwise.

    Args:
        dx: X component of the motion direction.
        dy: Y component of the motion direction.

    Returns:
        Heading in degrees within [0, 360).
    """
    return math.degrees(math.atan2(-dx, dy)) % 360.0


class RoamTrajectoryGenerator:
    """Generate smooth, collision-free roaming trajectories on a floorplan.

    The drivable region is the floor area minus furniture footprints. An
    occupancy grid and its distance transform give per-cell clearance; cells
    with clearance >= ``clearance`` form the navigable space. Spread waypoints
    are sampled within one connected navigable component, ordered by polar
    angle into a non-self-intersecting loop, and linked with a clearance-aware
    A* planner. Each segment is simplified inside the navigable region, dead-
    end hairpins are removed, corners are rounded, and a final clearance pass
    keeps every sample collision-free; the start and end are pushed into open
    space.

    Doors are treated as open passages by default (excluded from obstacles),
    matching scenes where doors are removed before rendering.
    """

    def __init__(
        self,
        floor: Geometry,
        obstacles: Geometry,
        clearance: float = DEFAULT_CLEARANCE,
        resolution: float = DEFAULT_RESOLUTION,
        rooms: dict[str, Geometry] | None = None,
    ) -> None:
        """Initialize the generator and build the navigability grid.

        Args:
            floor: Drivable floor region (e.g. union of all room floors).
            obstacles: Union of obstacle footprints (e.g. furniture).
            clearance: Minimum obstacle clearance (m) the path must keep.
            resolution: Occupancy-grid cell size in meters.
            rooms: Optional room-name to polygon map, used only to report
                which rooms the trajectory can reach.
        """
        if floor.is_empty:
            raise ValueError("Floor region is empty; cannot plan a path.")

        self.clearance = clearance
        self.resolution = resolution
        self.rooms = rooms or {}

        self.minx, self.miny, maxx, maxy = floor.bounds
        self.nx = max(1, int(math.ceil((maxx - self.minx) / resolution)))
        self.ny = max(1, int(math.ceil((maxy - self.miny) / resolution)))

        xs = self.minx + (np.arange(self.nx) + 0.5) * resolution
        ys = self.miny + (np.arange(self.ny) + 0.5) * resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        flat_x, flat_y = grid_x.ravel(), grid_y.ravel()

        inside = shapely.contains_xy(floor, flat_x, flat_y)
        if obstacles is not None and not obstacles.is_empty:
            in_obs = shapely.contains_xy(obstacles, flat_x, flat_y)
        else:
            in_obs = np.zeros_like(inside)
        free = (inside & ~in_obs).reshape(self.ny, self.nx)

        self.clearance_map = distance_transform_edt(free) * resolution
        self.nav = self.clearance_map >= clearance
        self.labels, n_comp = label(self.nav, structure=np.ones((3, 3)))
        logger.info(
            "Navigability grid %dx%d, %d component(s), max clearance %.2fm.",
            self.ny,
            self.nx,
            n_comp,
            float(self.clearance_map.max()),
        )

    @classmethod
    def from_collector(
        cls,
        collector,
        clearance: float = DEFAULT_CLEARANCE,
        resolution: float = DEFAULT_RESOLUTION,
        obstacle_ignore: tuple[str, ...] = DEFAULT_OBSTACLE_IGNORE,
        obstacle_clearance: float | None = None,
    ) -> "RoamTrajectoryGenerator":
        """Build a generator from a ``UrdfSemanticInfoCollector``.

        Args:
            collector: A collector that has already parsed a URDF scene.
            clearance: Minimum clearance (m) kept from walls (the floor
                boundary); keep it small so narrow doorways stay passable.
            resolution: Occupancy-grid cell size in meters.
            obstacle_ignore: Footprint name keywords to exclude from
                obstacles (treated as open passages, e.g. doors).
            obstacle_clearance: Minimum clearance (m) kept from furniture and
                objects. When larger than ``clearance``, footprints are
                inflated by the difference so the path stays this far from
                them while still only needing ``clearance`` from walls.
                Defaults to ``clearance`` (no extra furniture margin).

        Returns:
            A configured ``RoamTrajectoryGenerator``.
        """
        obstacle_polys = [
            poly
            for key, poly in collector.footprints.items()
            if not any(kw in key.lower() for kw in obstacle_ignore)
        ]
        obstacles = (
            unary_union(obstacle_polys) if obstacle_polys else Polygon()
        )
        margin = (obstacle_clearance or clearance) - clearance
        if margin > 0 and not obstacles.is_empty:
            obstacles = obstacles.buffer(margin)
        return cls(
            floor=collector.floor_union,
            obstacles=obstacles,
            clearance=clearance,
            resolution=resolution,
            rooms=collector.rooms,
        )

    # ----- coordinate helpers ------------------------------------------------

    def _to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (x, y) to grid (row, col)."""
        j = int((x - self.minx) / self.resolution)
        i = int((y - self.miny) / self.resolution)
        return i, j

    def _to_world(self, i: int, j: int) -> tuple[float, float]:
        """Convert grid (row, col) to world (x, y) at the cell center."""
        x = self.minx + (j + 0.5) * self.resolution
        y = self.miny + (i + 0.5) * self.resolution
        return x, y

    def _clearance_at(self, x: float, y: float) -> float:
        """Return obstacle clearance (m) at the cell nearest to (x, y)."""
        i, j = self._to_cell(x, y)
        if 0 <= i < self.ny and 0 <= j < self.nx:
            return float(self.clearance_map[i, j])
        return 0.0

    # ----- component / waypoint selection ------------------------------------

    def _select_component(
        self, start_xy: tuple[float, float] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose the navigable component to roam within.

        Args:
            start_xy: Optional world start position. The component containing
                the nearest navigable cell is used; otherwise the largest.

        Returns:
            Tuple of (component_mask, component_cells) where component_cells
            is an (M, 2) array of (row, col) indices.

        Raises:
            ValueError: If no navigable space exists.
        """
        nav_cells = np.argwhere(self.nav)
        if nav_cells.size == 0:
            raise ValueError(
                f"No navigable space at clearance {self.clearance}m. "
                "Lower --clearance or --resolution."
            )

        if start_xy is not None:
            i, j = self._to_cell(*start_xy)
            comp_id = (
                self.labels[i, j]
                if 0 <= i < self.ny and 0 <= j < self.nx
                else 0
            )
            if comp_id == 0:
                # Snap to the nearest navigable cell.
                deltas = nav_cells - np.array([i, j])
                nearest = nav_cells[np.argmin((deltas**2).sum(axis=1))]
                comp_id = self.labels[nearest[0], nearest[1]]
        else:
            counts = np.bincount(self.labels.ravel())
            counts[0] = 0  # background
            comp_id = int(counts.argmax())

        mask = self.labels == comp_id
        return mask, np.argwhere(mask)

    def _sample_waypoints(
        self,
        comp_cells: np.ndarray,
        num_waypoints: int,
        start_xy: tuple[float, float] | None,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Sample spread-out waypoints via farthest-point sampling.

        Args:
            comp_cells: (M, 2) array of navigable (row, col) indices.
            num_waypoints: Number of waypoints to select.
            start_xy: Optional world start; seeds the first waypoint.
            rng: Random generator for reproducibility.

        Returns:
            Ordered list of (row, col) waypoints, starting near ``start_xy``.
        """
        num_waypoints = max(2, min(num_waypoints, len(comp_cells)))

        if start_xy is not None:
            i, j = self._to_cell(*start_xy)
            first = int(
                np.argmin(((comp_cells - np.array([i, j])) ** 2).sum(axis=1))
            )
        else:
            first = int(rng.integers(len(comp_cells)))

        selected = [first]
        min_dist = ((comp_cells - comp_cells[first]) ** 2).sum(axis=1)
        for _ in range(num_waypoints - 1):
            nxt = int(min_dist.argmax())
            if min_dist[nxt] == 0:
                break
            selected.append(nxt)
            d = ((comp_cells - comp_cells[nxt]) ** 2).sum(axis=1)
            min_dist = np.minimum(min_dist, d)

        return [tuple(comp_cells[idx]) for idx in selected]

    @staticmethod
    def _order_waypoints(
        waypoints: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Order waypoints into a non-self-intersecting loop by polar angle.

        Sorting the waypoints by their angle around the centroid yields a
        roughly convex sweep, which avoids the tangled crossings and the
        collapsing back-and-forth of a nearest-neighbor tour, and keeps the
        roam length stable. The angularly-first waypoint becomes the start.

        Args:
            waypoints: List of (row, col) waypoints.

        Returns:
            Angularly ordered waypoints forming a clean roaming loop.
        """
        if len(waypoints) <= 2:
            return waypoints

        pts = np.array(waypoints, dtype=float)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
        order = np.argsort(angles)
        return [waypoints[int(i)] for i in order]

    def _snap_to_open(
        self,
        cell: tuple[int, int],
        mask: np.ndarray,
        min_clearance: float,
    ) -> tuple[int, int]:
        """Move a cell to the nearest navigable cell clear of walls/objects.

        Returns the navigable cell nearest to ``cell`` whose clearance is at
        least ``min_clearance`` (so the start/end sit comfortably away from
        obstacles). If the cell already qualifies it is kept; if no cell in the
        component qualifies, the original is returned with a warning.

        Args:
            cell: (row, col) to snap.
            mask: Boolean grid of the navigable component.
            min_clearance: Required minimum clearance (m) at the endpoint.

        Returns:
            A (row, col) cell with clearance >= ``min_clearance`` when one
            exists in the component, else the original cell.
        """
        i, j = cell
        if self.clearance_map[i, j] >= min_clearance:
            return cell
        open_cells = np.argwhere(mask & (self.clearance_map >= min_clearance))
        if open_cells.size == 0:
            logger.warning(
                "No cell with clearance >= %.2fm in the component; keeping "
                "endpoint as is.",
                min_clearance,
            )
            return cell
        deltas = open_cells - np.array([i, j])
        nearest = open_cells[(deltas**2).sum(axis=1).argmin()]
        return (int(nearest[0]), int(nearest[1]))

    def _remove_reversals(
        self,
        vertices: list[tuple[float, float]],
        mask: np.ndarray,
        min_dot: float = HAIRPIN_REVERSAL_DOT,
    ) -> list[tuple[float, float]]:
        """Collapse hairpin reversals in the routed polyline.

        A waypoint inside a dead-end (single-door) room makes the routed path
        enter and leave through the same doorway, an unroundable U-turn that
        waypoint-angle pruning cannot see. Here we work on the actual path
        vertices: any vertex whose incoming and outgoing directions nearly
        reverse is removed when the straight shortcut across it stays inside
        the navigable mask, peeling the dead-end excursion back to the door.

        Args:
            vertices: Routed-and-simplified polyline vertices.
            mask: Boolean grid of the navigable component.
            min_dot: Reversal threshold on the in/out direction dot product
                (−1 = full reversal); default −0.85 ≈ only near-180 deg
                hairpins, so normal roam turns in open space are preserved.

        Returns:
            Polyline with hairpin excursions collapsed.
        """
        pts = [np.array(v, dtype=float) for v in vertices]
        changed = True
        while changed and len(pts) > 2:
            changed = False
            for i in range(1, len(pts) - 1):
                d_in = pts[i] - pts[i - 1]
                d_out = pts[i + 1] - pts[i]
                n_in = float(np.hypot(d_in[0], d_in[1]))
                n_out = float(np.hypot(d_out[0], d_out[1]))
                if n_in < 1e-9 or n_out < 1e-9:
                    pts.pop(i)
                    changed = True
                    break
                dot = float(np.dot(d_in / n_in, d_out / n_out))
                if dot < min_dot and self._segment_in_mask(
                    (pts[i - 1][0], pts[i - 1][1]),
                    (pts[i + 1][0], pts[i + 1][1]),
                    mask,
                ):
                    pts.pop(i)
                    changed = True
                    break
        return [(float(p[0]), float(p[1])) for p in pts]

    # ----- A* planning -------------------------------------------------------

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        mask: np.ndarray,
        center_weight: float = 2.0,
    ) -> list[tuple[int, int]] | None:
        """Plan a path between two cells on the navigable component.

        The step cost is scaled up where clearance is low, so the planner
        prefers well-centered routes away from walls and furniture.

        Args:
            start: Start (row, col).
            goal: Goal (row, col).
            mask: Boolean grid of the navigable component.
            center_weight: Strength of the centering preference.

        Returns:
            List of (row, col) cells from start to goal, or None if no path.
        """
        if start == goal:
            return [start]

        target = max(self.clearance * 2.0, self.clearance + 0.1)
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        g_score = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current)

            ci, cj = current
            for di, dj, step in _NEIGHBORS:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.ny and 0 <= nj < self.nx):
                    continue
                if not mask[ni, nj]:
                    continue
                deficit = max(0.0, target - self.clearance_map[ni, nj])
                cost = step * (1.0 + center_weight * deficit / target)
                tentative = g_score[current] + cost
                if tentative < g_score.get((ni, nj), math.inf):
                    g_score[(ni, nj)] = tentative
                    came_from[(ni, nj)] = current
                    h = math.hypot(goal[0] - ni, goal[1] - nj)
                    heapq.heappush(open_heap, (tentative + h, (ni, nj)))

        return None

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Rebuild a path from the A* came-from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ----- smoothing ---------------------------------------------------------

    def _segment_in_mask(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        mask: np.ndarray,
    ) -> bool:
        """Check whether the straight segment a-b stays within the mask."""
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(dist / (self.resolution * 0.5)))
        for t in np.linspace(0.0, 1.0, steps):
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            i, j = self._to_cell(x, y)
            if not (0 <= i < self.ny and 0 <= j < self.nx and mask[i, j]):
                return False
        return True

    def _simplify_in_mask(
        self,
        points: list[tuple[float, float]],
        mask: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Greedily drop points while each chord stays inside the mask.

        Removes the per-cell A* staircase so straight runs collapse to their
        endpoints, but keeps a vertex wherever skipping it would let a chord
        leave the navigable region (i.e. cut through a wall). Endpoints are
        always kept. Applied per waypoint segment, so the roam shape (the
        sequence of waypoints) is preserved.

        Args:
            points: Dense polyline vertices of one A* segment.
            mask: Boolean grid of the navigable component.

        Returns:
            Simplified vertices whose consecutive chords stay in the mask.
        """
        if len(points) <= 2:
            return list(points)
        kept = [points[0]]
        anchor = 0
        for i in range(1, len(points) - 1):
            if not self._segment_in_mask(points[anchor], points[i + 1], mask):
                kept.append(points[i])
                anchor = i
        kept.append(points[-1])
        return kept

    def _round_corners(
        self,
        points: list[tuple[float, float]],
        radius: float,
        samples_per_corner: int = 12,
    ) -> list[tuple[float, float]]:
        """Round each turn with a clearance-safe arc of a target radius.

        At every interior vertex a quadratic Bezier fillet is inserted whose
        offset from the corner equals ``radius`` (clamped to half of each
        adjacent segment). A larger ``radius`` yields wider turning arcs.
        A corner is only rounded when the entire arc keeps the required
        clearance; otherwise the sharp vertex is kept.

        Args:
            points: Polyline vertices.
            radius: Target turn radius (m); larger gives wider arcs.
            samples_per_corner: Points sampled along each rounded corner.

        Returns:
            Polyline vertices with rounded turns.
        """
        if radius <= 0 or len(points) <= 2:
            return points

        out: list[tuple[float, float]] = [points[0]]
        for k in range(1, len(points) - 1):
            a = np.asarray(points[k - 1], dtype=float)
            v = np.asarray(points[k], dtype=float)
            b = np.asarray(points[k + 1], dtype=float)
            va, vb = a - v, b - v
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            if la < 1e-6 or lb < 1e-6:
                out.append((float(v[0]), float(v[1])))
                continue

            # Round with the largest radius that stays clear; shrink and retry
            # so constrained corners still get a (smaller) arc rather than a
            # sharp cusp that snaps the heading.
            r = min(radius, la * 0.5, lb * 0.5)
            arc: list[tuple[float, float]] = []
            while r >= MIN_FILLET_RADIUS:
                p1 = v + va / la * r
                p2 = v + vb / lb * r
                n_samples = max(
                    samples_per_corner, int(2 * r / self.resolution)
                )
                candidate = []
                safe = True
                for t in np.linspace(0.0, 1.0, n_samples):
                    pt = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * v + t**2 * p2
                    if self._clearance_at(pt[0], pt[1]) < self.clearance:
                        safe = False
                        break
                    candidate.append((float(pt[0]), float(pt[1])))
                if safe:
                    arc = candidate
                    break
                r *= 0.6
            out.extend(arc if arc else [(float(v[0]), float(v[1]))])

        out.append(points[-1])
        return out

    def _resample(
        self, points: list[tuple[float, float]], spacing: float
    ) -> np.ndarray:
        """Resample a polyline to uniform arc-length spacing (constant speed).

        Args:
            points: Polyline vertices.
            spacing: Spacing (m) between consecutive points.

        Returns:
            (M, 2) array of equidistant points including both endpoints.
        """
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total <= spacing:
            return pts
        n_intervals = max(1, round(total / spacing))
        targets = np.linspace(0.0, total, n_intervals + 1)
        x = np.interp(targets, cum, pts[:, 0])
        y = np.interp(targets, cum, pts[:, 1])
        return np.column_stack([x, y])

    def _clearance_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Finite-difference gradient of the clearance field at (x, y)."""
        step = self.resolution
        gx = self._clearance_at(x + step, y) - self._clearance_at(x - step, y)
        gy = self._clearance_at(x, y + step) - self._clearance_at(x, y - step)
        return gx, gy

    def _enforce_clearance(
        self, xy: np.ndarray, max_steps: int = 40
    ) -> np.ndarray:
        """Push any sample below the clearance bound up the clearance field.

        Smoothing and linear resampling can let a point dip below the
        required clearance near concave corners; this nudges such points
        toward higher clearance until the bound is satisfied.

        Args:
            xy: (N, 2) array of waypoints.
            max_steps: Maximum gradient-ascent steps per point.

        Returns:
            The adjusted (N, 2) array.
        """
        out = xy.copy()
        for k in range(len(out)):
            x, y = out[k]
            for _ in range(max_steps):
                if self._clearance_at(x, y) >= self.clearance:
                    break
                gx, gy = self._clearance_gradient(x, y)
                norm = math.hypot(gx, gy)
                if norm < 1e-9:
                    break
                x += self.resolution * gx / norm
                y += self.resolution * gy / norm
            out[k] = (x, y)
        return out

    @staticmethod
    def _compute_headings(xy: np.ndarray) -> np.ndarray:
        """Compute per-point heading (rot_deg) tangent to the curve."""
        n = len(xy)
        rots = np.zeros(n)
        for i in range(n):
            if i == 0:
                dx, dy = xy[1] - xy[0]
            elif i == n - 1:
                dx, dy = xy[-1] - xy[-2]
            else:
                dx, dy = xy[i + 1] - xy[i - 1]
            rots[i] = heading_to_rot_deg(dx, dy)
        return rots

    @staticmethod
    def _limit_heading_rate(
        points: np.ndarray,
        trigger_deg: float = TURN_FRAME_TRIGGER_DEG,
        step_deg: float = TURN_FRAME_STEP_DEG,
    ) -> np.ndarray:
        """Add in-place rotation frames only at big U-turns.

        Gentle turns flow continuously; only where the heading jumps more than
        ``trigger_deg`` between consecutive points (an unavoidable sharp corner)
        does the robot rotate in place, with a few extra frames stepping the
        heading by ``step_deg`` so the big turn is not abrupt.

        Args:
            points: (N, 3) array of (x, y, rot_deg).
            trigger_deg: Only jumps above this get extra frames.
            step_deg: Heading step (deg) of each inserted in-place frame.

        Returns:
            (M, 3) array with M >= N.
        """
        if len(points) < 2:
            return points
        out = [points[0]]
        for i in range(1, len(points)):
            prev_rot = out[-1][2]
            x, y, rot = points[i]
            delta = (rot - prev_rot + 180.0) % 360.0 - 180.0
            if abs(delta) > trigger_deg:
                n_extra = max(1, int(math.ceil(abs(delta) / step_deg)) - 1)
                for k in range(1, n_extra + 1):
                    out.append(
                        (x, y, (prev_rot + delta * k / (n_extra + 1)) % 360.0)
                    )
            out.append((x, y, rot))
        return np.asarray(out)

    # ----- public API --------------------------------------------------------

    def generate(
        self,
        start_xy: tuple[float, float] | None = None,
        num_waypoints: int = DEFAULT_NUM_WAYPOINTS,
        point_spacing: float = DEFAULT_POINT_SPACING,
        turn_radius: float = DEFAULT_TURN_RADIUS,
        endpoint_clearance: float = DEFAULT_ENDPOINT_CLEARANCE,
        seed: int | None = None,
    ) -> TrajectoryResult:
        """Generate a smooth roaming trajectory.

        Args:
            start_xy: Optional world (x, y) start. Defaults to a point in the
                largest navigable component.
            num_waypoints: Number of roam waypoints to visit.
            point_spacing: Output sampling spacing in meters.
            turn_radius: Target turning-arc radius (m); larger gives wider,
                rounder turns (clamped by segment length and clearance).
            endpoint_clearance: Minimum clearance (m) kept at the start and
                end points so they are not placed close to walls.
            seed: Optional RNG seed for reproducible roaming.

        Returns:
            A ``TrajectoryResult`` with (x, y, rot_deg) waypoints.

        Raises:
            ValueError: If no navigable path can be planned.
        """
        rng = np.random.default_rng(seed)
        mask, comp_cells = self._select_component(start_xy)

        waypoints = self._sample_waypoints(
            comp_cells, num_waypoints, start_xy, rng
        )
        waypoints = self._order_waypoints(waypoints)

        # Keep the end (and the auto-chosen start) well away from walls.
        waypoints[-1] = self._snap_to_open(
            waypoints[-1], mask, endpoint_clearance
        )
        if start_xy is None:
            waypoints[0] = self._snap_to_open(
                waypoints[0], mask, endpoint_clearance
            )

        # Plan each waypoint-to-waypoint segment and simplify it within the
        # navigable mask. Simplifying per segment (not the merged path) keeps
        # every chord inside walls AND preserves the waypoints (roam shape),
        # while collapsing straight runs so corners can use the full radius.
        vertices: list[tuple[float, float]] = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            segment = self._astar(a, b, mask)
            if segment is None:
                logger.warning("No A* path between %s and %s; skipping.", a, b)
                continue
            seg_world = [self._to_world(i, j) for i, j in segment]
            seg_world = self._simplify_in_mask(seg_world, mask)
            if vertices and seg_world and vertices[-1] == seg_world[0]:
                seg_world = seg_world[1:]
            vertices.extend(seg_world)

        if len(vertices) < 2:
            raise ValueError(
                "Failed to plan a roaming path; try more waypoints or a "
                "lower clearance."
            )

        # Collapse dead-end hairpins (single-door room excursions) that the
        # waypoint-angle prune cannot detect.
        vertices = self._remove_reversals(vertices, mask)

        world_path = self._round_corners(vertices, turn_radius)
        # Resample to uniform spacing, then enforce clearance last so the
        # output is collision-free (the final interpolation cannot reintroduce
        # violations). Spacing stays near-uniform; only the few points pushed
        # out near tight corners shift, and gen_trajectory derives timestamps
        # from arc length so constant speed is preserved.
        xy = self._resample(world_path, point_spacing)
        xy = self._enforce_clearance(xy)

        rots = self._compute_headings(xy)
        points = np.column_stack([xy, rots])
        # Insert in-place rotations at unavoidably sharp corners so the
        # heading never snaps between consecutive points.
        points = self._limit_heading_rate(points)

        xy = points[:, :2]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = float(seg.sum())
        min_clr = min(self._clearance_at(x, y) for x, y in xy)
        reachable = self._reachable_rooms(mask)

        logger.info(
            "Generated trajectory: %d points, %.2fm long, min clearance "
            "%.2fm, reaches %d room(s).",
            len(points),
            length,
            min_clr,
            len(reachable),
        )
        return TrajectoryResult(
            points=points,
            clearance=self.clearance,
            min_clearance=min_clr,
            length=length,
            reachable_rooms=reachable,
        )

    def _reachable_rooms(self, mask: np.ndarray) -> list[str]:
        """List room keys overlapping the chosen navigable component.

        A room is reachable if any navigable cell of the component falls
        inside the room polygon.

        Args:
            mask: Boolean grid of the navigable component.

        Returns:
            Room keys with at least one navigable cell inside them.
        """
        cells = np.argwhere(mask)
        if cells.size == 0:
            return []
        world = np.array([self._to_world(i, j) for i, j in cells])
        reachable = []
        for name, poly in self.rooms.items():
            if poly.is_empty:
                continue
            if shapely.contains_xy(poly, world[:, 0], world[:, 1]).any():
                reachable.append(name)
        return reachable
