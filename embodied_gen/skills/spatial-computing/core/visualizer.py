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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# Type aliases
Geometry = Polygon | MultiPolygon

logger = logging.getLogger(__name__)


class FloorplanVisualizer:
    """Static utility class for visualizing floorplans."""

    @staticmethod
    def draw_poly(ax: Axes, poly: Geometry, **kwargs) -> None:
        """Draw a polygon or multi-polygon on matplotlib axes.

        Args:
            ax: Matplotlib axes object.
            poly: Shapely Polygon or MultiPolygon to draw.
            **kwargs: Additional arguments passed to ax.fill().

        """
        if poly.is_empty:
            return

        geoms = poly.geoms if hasattr(poly, "geoms") else [poly]

        color = kwargs.pop("color", None)
        if color is None:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(geoms))]
        else:
            colors = [color] * len(geoms)

        for i, p in enumerate(geoms):
            if p.is_empty:
                continue
            x, y = p.exterior.xy
            ax.fill(x, y, facecolor=colors[i], **kwargs)

    @classmethod
    def plot(
        cls,
        rooms: dict[str, Geometry],
        footprints: dict[str, Geometry],
        occ_area: Geometry,
        save_path: str,
        trajectory: np.ndarray | None = None,
        arrow_stride: int = 10,
        current_index: int | None = None,
        point_markers: bool = True,
        dpi: int = 300,
    ) -> None:
        """Generate and save a floorplan visualization.

        Args:
            rooms: Dictionary mapping room names to floor polygons.
            footprints: Dictionary mapping object names to footprint polygons.
            occ_area: Union of all occupied areas.
            save_path: Path to save the output image.
            trajectory: Optional (N, 2) or (N, 3) array of waypoints. When the
                third column (rot_deg, tangent heading) is present, heading
                arrows are drawn. Rendered as a red curve overlay.
            arrow_stride: Draw a heading arrow every ``arrow_stride`` points
                (0 disables arrows). Ignored when ``current_index`` is set.
            current_index: Animation frame index. When set, only the traveled
                path (up to this index) is drawn, with a green dot at the
                current position and a red heading arrow; the future path is
                hidden.
            point_markers: When True, mark every trajectory point with a small
                red dot (in addition to the curve).
            dpi: Output image resolution in dots per inch.

        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        cmap_rooms = plt.get_cmap("Pastel1")

        cls._draw_room_floors(ax, rooms, cmap_rooms)
        cls._draw_occupied_area(ax, occ_area)
        cls._draw_footprint_outlines(ax, footprints)
        cls._draw_footprint_labels(ax, footprints)
        cls._draw_room_labels(ax, rooms)
        if trajectory is not None and len(trajectory) > 1:
            cls._draw_trajectory(
                ax,
                np.asarray(trajectory),
                arrow_stride,
                current_index,
                point_markers,
            )
        cls._configure_axes(ax, rooms, occ_area)

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig)

    @classmethod
    def _draw_room_floors(
        cls,
        ax: Axes,
        rooms: dict[str, Geometry],
        cmap: plt.cm.ScalarMappable,
    ) -> None:
        """Draw colored room floor polygons (Layer 1)."""
        for i, (name, poly) in enumerate(rooms.items()):
            color = cmap(i % cmap.N)
            cls.draw_poly(
                ax,
                poly,
                color=color,
                alpha=1.0,
                edgecolor="black",
                linestyle="--",
                zorder=1,
            )

    @classmethod
    def _draw_occupied_area(cls, ax: Axes, occ_area: Geometry) -> None:
        """Draw the occupied area overlay (Layer 2)."""
        cls.draw_poly(
            ax,
            occ_area,
            color="tab:blue",
            alpha=0.5,
            lw=0,
            zorder=2,
        )

    @staticmethod
    def _draw_footprint_outlines(
        ax: Axes,
        footprints: dict[str, Geometry],
    ) -> None:
        """Draw footprint outlines (Layer 3)."""
        for poly in footprints.values():
            if poly.is_empty:
                continue
            geoms = poly.geoms if hasattr(poly, "geoms") else [poly]
            for p in geoms:
                ax.plot(*p.exterior.xy, "--", lw=0.8, color="gray", zorder=3)

    @staticmethod
    def _draw_footprint_labels(
        ax: Axes,
        footprints: dict[str, Geometry],
    ) -> None:
        """Draw footprint text labels (Layer 4)."""
        import re

        for name, poly in footprints.items():
            if poly.is_empty:
                continue
            label = re.sub(r"_\d+$", "", name)
            ax.text(
                poly.centroid.x,
                poly.centroid.y,
                label,
                fontsize=8,
                ha="center",
                va="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.5,
                    "edgecolor": "none",
                    "pad": 0.1,
                },
                zorder=4,
            )

    @staticmethod
    def _draw_room_labels(ax: Axes, rooms: dict[str, Geometry]) -> None:
        """Draw room text labels (Layer 5)."""
        for name, poly in rooms.items():
            if poly.is_empty:
                continue
            label = name.replace("_floor", "")
            ax.text(
                poly.centroid.x,
                poly.centroid.y,
                label,
                fontsize=9,
                color="black",
                weight="bold",
                ha="center",
                va="center",
                bbox={
                    "facecolor": "lightgray",
                    "alpha": 0.7,
                    "edgecolor": "black",
                    "boxstyle": "round,pad=0.3",
                },
                zorder=5,
            )

    @staticmethod
    def _draw_trajectory(
        ax: Axes,
        trajectory: np.ndarray,
        arrow_stride: int,
        current_index: int | None = None,
        point_markers: bool = True,
    ) -> None:
        """Draw the roaming trajectory as a red curve with heading arrows.

        Args:
            ax: Matplotlib axes object.
            trajectory: (N, 2) or (N, 3) array; the optional third column is
                the heading rot_deg (0 = +Y, counter-clockwise positive).
            arrow_stride: Spacing (in points) between heading arrows; 0 to
                disable arrows. Ignored when ``current_index`` is set.
            current_index: When set, draw only the traveled path up to this
                index, a green dot at the current position, and a single red
                heading arrow; the future path is hidden.
            point_markers: When True, mark every trajectory point with a small
                red dot (full-trajectory view only).

        """
        xs, ys = trajectory[:, 0], trajectory[:, 1]
        has_heading = trajectory.shape[1] >= 3

        if current_index is not None:
            ci = int(np.clip(current_index, 0, len(trajectory) - 1))
            ax.plot(
                xs[: ci + 1],
                ys[: ci + 1],
                color="red",
                linewidth=2.5,
                solid_capstyle="round",
                zorder=6,
            )
            ax.scatter(
                xs[ci], ys[ci], c="lime", s=200, edgecolors="black", zorder=8
            )
            if has_heading:
                rot = np.deg2rad(trajectory[ci, 2])
                dx, dy = -np.sin(rot), np.cos(rot)
                # Tail anchored at the dot center; long arrow in data coords.
                length = 0.9
                head = (xs[ci] + dx * length, ys[ci] + dy * length)
                ax.annotate(
                    "",
                    xy=head,
                    xytext=(xs[ci], ys[ci]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "red",
                        "lw": 2.5,
                        "mutation_scale": 18,
                    },
                    zorder=9,
                )
            return

        ax.plot(
            xs,
            ys,
            color="red",
            linewidth=2.5,
            solid_capstyle="round",
            zorder=6,
        )
        if point_markers:
            ax.scatter(xs, ys, c="gray", s=8, zorder=6.5, edgecolors="none")
        ax.scatter(
            xs[0],
            ys[0],
            c="lime",
            s=70,
            edgecolors="black",
            zorder=7,
            label="start",
        )
        ax.scatter(
            xs[-1],
            ys[-1],
            c="red",
            s=70,
            marker="s",
            edgecolors="black",
            zorder=7,
            label="end",
        )

        if arrow_stride and has_heading:
            idx = np.arange(0, len(trajectory), arrow_stride)
            rot = np.deg2rad(trajectory[idx, 2])
            # Heading convention: 0 deg -> +Y, counter-clockwise positive.
            u, v = -np.sin(rot), np.cos(rot)
            ax.quiver(
                xs[idx],
                ys[idx],
                u,
                v,
                color="darkred",
                scale=25,
                width=0.004,
                zorder=7,
            )

    @staticmethod
    def _configure_axes(
        ax: Axes,
        rooms: dict[str, Geometry],
        occ_area: Geometry,
    ) -> None:
        """Configure axes limits and labels."""
        total_geom = unary_union(list(rooms.values()) + [occ_area])

        if total_geom.is_empty:
            minx, miny, maxx, maxy = -1, -1, 1, 1
        else:
            minx, miny, maxx, maxy = total_geom.bounds

        cx = (minx + maxx) * 0.5
        cy = (miny + maxy) * 0.5
        half = max(maxx - minx, maxy - miny) * 0.5 * 1.05

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title("Floorplan Analysis", fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
