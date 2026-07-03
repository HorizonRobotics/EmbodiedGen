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
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from shapely.geometry import MultiPolygon, Polygon
from embodied_gen.utils.llm_resolve import resolve_instance_with_llm

from ..core import (
    UrdfSemanticInfoCollector,
)
from ..core.collector import (
    DEFAULT_BESIDE_DISTANCE,
    DEFAULT_IGNORE_ITEMS,
    DEFAULT_MESH_SAMPLE_NUM,
    DEFAULT_ROTATION_RPY,
)
from ..core.visualizer import (
    FloorplanVisualizer,
)

# Type aliases
Geometry = Polygon | MultiPolygon
logger = logging.getLogger(__name__)


@dataclass
class FloorplanConfig:
    """Configuration for floorplan operations."""

    urdf_path: str
    """Path to the input URDF scene file."""

    output_path: str | None = None
    """Path to save the floorplan visualization image."""

    usd_path: str | None = None
    """Optional path to the USD scene file for USD export."""

    asset_path: str | None = None
    """Optional path to the asset mesh file (.obj)."""

    instance_key: str = "inserted_object"
    """Unique key for the added instance."""

    in_room: str | None = None
    """Optional room name to constrain asset placement."""

    on_instance: str | None = None
    """Optional instance name to place the asset on top of (exact key from get_instance_names())."""

    beside_instance: str | None = None
    """Optional instance name to place the asset beside (on floor, near the target)."""

    beside_distance: float = DEFAULT_BESIDE_DISTANCE
    """Max distance (meters) from the target instance for beside placement."""

    place_strategy: Literal["top", "random"] = "random"
    """Placement strategy for the asset."""

    rotation_rpy: tuple[float, float, float] = DEFAULT_ROTATION_RPY
    """Rotation in roll-pitch-yaw (radians)."""

    ignore_items: list[str] = field(
        default_factory=lambda: list(DEFAULT_IGNORE_ITEMS)
    )
    """List of item name patterns to ignore during parsing."""

    mesh_sample_num: int = DEFAULT_MESH_SAMPLE_NUM
    """Number of points to sample from meshes."""

    max_placement_attempts: int = 2000
    """Maximum attempts for asset placement."""

    update_urdf: bool = True
    """Whether to update and save the URDF file."""

    update_usd: bool = True
    """Whether to update and save the USD file."""

    list_instances: bool = False
    """If True, print instance and room names then exit (no placement/visualization)."""

    delete_instance: str | None = None
    """Optional instance name to delete from the scene (supports fuzzy matching with LLM)."""

    delete_in_room: str | None = None
    """Optional room constraint for deletion - only delete if instance is in this room."""

    query_instance: str | None = None
    """Optional instance name to query and return its center coordinates (supports fuzzy matching with LLM)."""

    output_strategy: Literal["suffix", "overwrite", "timestamp"] = "suffix"
    """File naming strategy for output files.

    - "suffix": Add '_updated' suffix (default, non-destructive)
    - "overwrite": Overwrite original files (use with caution)
    - "timestamp": Add timestamp suffix (e.g., '_20260311_171500')
    """

    batch_insert_config: str | None = None
    """Path to JSON config file for batch insertion (3-4x faster than multiple CLI calls).

    JSON format example:
    [
        {
            "asset_path": "path/to/chair1.obj",
            "instance_key": "chair_1",
            "beside_instance": "table_dining_7178300",
            "in_room": "dining_room_0_floor"
        },
        {
            "asset_path": "path/to/chair2.obj",
            "instance_key": "chair_2",
            "beside_instance": "table_dining_7178300",
            "in_room": "dining_room_0_floor"
        }
    ]
    """


class FloorplanManager:
    """High-level API for floorplan operations.

    This class provides simplified methods for:
    - Loading and analyzing URDF scenes
    - Visualizing floorplans
    - Inserting objects into scenes
    - Updating URDF and USD files

    Example:
        >>> manager = FloorplanManager(urdf_path="scene.urdf", usd_path="scene.usdc")
        >>> manager.visualize(output_path="floorplan.png")
        >>> position = manager.insert_object(
        ...     asset_path="chair.obj",
        ...     instance_key="chair_1",
        ...     in_room="kitchen"
        ... )
        # URDF/USD are updated automatically after insert
    """

    def __init__(
        self,
        urdf_path: str,
        usd_path: str | None = None,
        mesh_sample_num: int = DEFAULT_MESH_SAMPLE_NUM,
        ignore_items: list[str] | None = None,
        output_strategy: Literal[
            "suffix", "overwrite", "timestamp"
        ] = "suffix",
    ) -> None:
        """Initialize the floorplan manager.

        Args:
            urdf_path: Path to the URDF file.
            usd_path: Optional path to the USD file for scene updates.
            mesh_sample_num: Number of points to sample from meshes.
            ignore_items: List of item name patterns to ignore.
            output_strategy: File naming strategy for output files.

        """
        self.urdf_path = urdf_path
        self.usd_path = usd_path
        self.output_strategy = output_strategy
        self.collector = UrdfSemanticInfoCollector(
            mesh_sample_num=mesh_sample_num,
            ignore_items=ignore_items,
        )
        self.collector.collect(urdf_path)
        self.pending_instance_data: dict | None = None

    def _get_output_path(
        self,
        input_path: str,
        custom_output_path: str | None = None,
    ) -> str:
        """Generate output path based on the naming strategy.

        Smart file naming strategy:
        - "suffix" (default):
          * If input already ends with "_updated", overwrite it (continuous operations)
          * Otherwise, add "_updated" suffix (first operation)
        - "timestamp": Add timestamp suffix for unique versioning
        - "overwrite": Always overwrite the input file

        Args:
            input_path: Original input file path.
            custom_output_path: Optional custom output path (highest priority).

        Returns:
            Generated output path based on strategy.

        """
        if custom_output_path is not None:
            return custom_output_path

        name, ext = os.path.splitext(input_path)

        if self.output_strategy == "overwrite":
            return input_path
        elif self.output_strategy == "timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{name}_{timestamp}{ext}"
        else:  # "suffix" (default) - smart continuous operation support
            # If input already has "_updated" suffix, overwrite it (continuous operation)
            if name.endswith("_updated"):
                return input_path
            # Otherwise, add "_updated" suffix (first operation)
            else:
                return f"{name}_updated{ext}"

    def visualize(
        self,
        output_path: str,
    ) -> None:
        """Generate and save a floorplan visualization.

        Args:
            output_path: Path to save the output image.

        """
        FloorplanVisualizer.plot(
            self.collector.rooms,
            self.collector.footprints,
            self.collector.occ_area,
            output_path,
        )
        logger.info(f"✅ Floorplan visualization saved to {output_path}")

    def insert_object(
        self,
        asset_path: str,
        instance_key: str,
        in_room: str | None = None,
        on_instance: str | None = None,
        beside_instance: str | None = None,
        beside_distance: float = DEFAULT_BESIDE_DISTANCE,
        rotation_rpy: tuple[float, float, float] = DEFAULT_ROTATION_RPY,
        n_max_attempt: int = 2000,
        place_strategy: Literal["top", "random"] = "random",
    ) -> list[float] | None:
        """Insert an object into the scene with automatic placement.

        Args:
            asset_path: Path to the asset mesh file (.obj).
            instance_key: Unique key for the new instance.
            in_room: Optional room name to constrain placement.
            on_instance: Optional instance name to place on top of.
            beside_instance: Optional instance name to place beside (on floor).
            beside_distance: Max distance from target for beside placement.
            rotation_rpy: Initial rotation in roll-pitch-yaw.
            n_max_attempt: Maximum placement attempts.
            place_strategy: Either "top" or "random".

        Returns:
            List [x, y, z] of the placed instance center, or None if failed.

        """
        center = self.collector.add_instance(
            asset_path=asset_path,
            instance_key=instance_key,
            in_room=in_room,
            on_instance=on_instance,
            beside_instance=beside_instance,
            beside_distance=beside_distance,
            rotation_rpy=rotation_rpy,
            n_max_attempt=n_max_attempt,
            place_strategy=place_strategy,
        )

        if center is not None:
            self.pending_instance_data = {
                "asset_path": asset_path,
                "instance_key": instance_key,
                "center": center,
                "rotation_rpy": rotation_rpy,
            }
            self.update_scene()

        return center

    def batch_insert_objects(
        self,
        objects: list[dict],
        defer_update: bool = False,
    ) -> list[list[float] | None]:
        """Batch insert multiple objects into the scene efficiently.

        Args:
            objects: List of object configs, each containing:
                asset_path: Path to the asset mesh file (.obj).
                instance_key: Unique key for the new instance.
                in_room: Optional room name to constrain placement.
                on_instance: Optional instance name to place on top of.
                beside_instance: Optional instance name to place beside.
                beside_distance: Max distance from target (default: 0.5m).
                rotation_rpy: Initial rotation (default: (0, 0, 0)).
                place_strategy: Either "top" or "random" (default: "random").
            defer_update: If True, don't update URDF/USD after each
                insertion. Useful when inserting many objects at once.

        Returns:
            List of centers [x, y, z] for each inserted object,
            or None if failed.

        Example:
            >>> objects = [
            ...     {"asset_path": "chair1.obj",
            ...      "instance_key": "chair_1",
            ...      "beside_instance": "table"},
            ... ]
            >>> centers = manager.batch_insert_objects(objects)

        """
        centers = []
        usd_source = self.usd_path

        for i, obj_config in enumerate(objects, 1):
            logger.info(
                f"[{i}/{len(objects)}] Inserting '{obj_config.get('instance_key', 'unknown')}'..."
            )

            center = self.collector.add_instance(
                asset_path=obj_config["asset_path"],
                instance_key=obj_config["instance_key"],
                in_room=obj_config.get("in_room"),
                on_instance=obj_config.get("on_instance"),
                beside_instance=obj_config.get("beside_instance"),
                beside_distance=obj_config.get(
                    "beside_distance", DEFAULT_BESIDE_DISTANCE
                ),
                rotation_rpy=obj_config.get(
                    "rotation_rpy", DEFAULT_ROTATION_RPY
                ),
                n_max_attempt=obj_config.get("n_max_attempt", 2000),
                place_strategy=obj_config.get("place_strategy", "random"),
            )

            if center is not None:
                # Store instance data for later update
                collision_path = obj_config["asset_path"].replace(
                    ".obj", "_collision.obj"
                )
                if not os.path.exists(collision_path):
                    collision_path = None

                # Update URDF incrementally
                if self.urdf_path and not defer_update:
                    urdf_out = self._get_output_path(self.urdf_path)
                    self.collector.update_urdf_info(
                        output_path=urdf_out,
                        instance_key=obj_config["instance_key"],
                        visual_mesh_path=obj_config["asset_path"],
                        collision_mesh_path=collision_path,
                        trans_xyz=tuple(center),
                        rot_rpy=obj_config.get(
                            "rotation_rpy", DEFAULT_ROTATION_RPY
                        ),
                        joint_type="fixed",
                    )

                # Update USD incrementally
                if self.usd_path and not defer_update:
                    usd_out = self._get_output_path(self.usd_path)
                    self.collector.update_usd_info(
                        usd_path=usd_source,
                        output_path=usd_out,
                        instance_key=obj_config["instance_key"],
                        visual_mesh_path=obj_config["asset_path"],
                        trans_xyz=center,
                        rot_rpy=obj_config.get(
                            "rotation_rpy", DEFAULT_ROTATION_RPY
                        ),
                    )
                    usd_source = usd_out

                logger.info(f"✅ Placed at {center}")
            else:
                logger.warning(f"❌ Failed to place object")

            centers.append(center)

        return centers

    def update_scene(
        self,
        urdf_output_path: str | None = None,
        usd_output_path: str | None = None,
    ) -> None:
        """Update URDF and/or USD with inserted instances.

        Updates URDF if self.urdf_path is set, USD if self.usd_path is set.
        Both are updated when both paths are set. No-op when no instance was inserted.

        Note: USD updates require Blender (bpy) to convert .obj to .usdc format.

        Args:
            urdf_output_path: Optional custom path for URDF output.
            usd_output_path: Optional custom path for USD output.

        Raises:
            ValueError: If no instance has been inserted.

        """
        if self.pending_instance_data is None:
            raise ValueError(
                "No instance to update. Call insert_object() first."
            )

        data = self.pending_instance_data
        collision_path = data["asset_path"].replace(".obj", "_collision.obj")
        if not os.path.exists(collision_path):
            collision_path = None

        if self.urdf_path:
            urdf_out = self._get_output_path(self.urdf_path, urdf_output_path)
            self.collector.update_urdf_info(
                output_path=urdf_out,
                instance_key=data["instance_key"],
                visual_mesh_path=data["asset_path"],
                collision_mesh_path=collision_path,
                trans_xyz=tuple(data["center"]),
                rot_rpy=data["rotation_rpy"],
                joint_type="fixed",
            )

        if self.usd_path:
            usd_out = self._get_output_path(self.usd_path, usd_output_path)
            self.collector.update_usd_info(
                usd_path=self.usd_path,
                output_path=usd_out,
                instance_key=data["instance_key"],
                visual_mesh_path=data["asset_path"],
                trans_xyz=data["center"],
                rot_rpy=data["rotation_rpy"],
            )

    def delete_object(
        self,
        instance_key: str,
        in_room: str | None = None,
        urdf_output_path: str | None = None,
        usd_output_path: str | None = None,
    ) -> bool:
        """Delete an object from the scene.

        Args:
            instance_key: Exact instance name to delete.
            in_room: Optional room constraint - only delete if instance is in this room.
            urdf_output_path: Optional custom path for URDF output.
            usd_output_path: Optional custom path for USD output.

        Returns:
            True if deletion succeeded, False otherwise.

        """
        success = self.collector.remove_instance(
            instance_key=instance_key,
            in_room=in_room,
        )

        if success:
            # Update URDF
            if self.urdf_path:
                urdf_out = self._get_output_path(
                    self.urdf_path, urdf_output_path
                )
                self.collector.save_urdf(urdf_out)

            # Update USD
            if self.usd_path:
                usd_out = self._get_output_path(self.usd_path, usd_output_path)
                self.collector.remove_usd_instance(
                    usd_path=self.usd_path,
                    output_path=usd_out,
                    instance_key=instance_key,
                )

        return success

    def get_instance_names(self) -> list[str]:
        """Get list of instance names in the scene.

        Returns:
            List of instance names.

        """
        return [
            k
            for k in self.collector.instances.keys()
            if k != "walls" and "floor" not in k.lower()
        ]

    def get_room_names(self) -> list[str]:
        """Get list of room names in the scene.

        Returns:
            List of room names.

        """
        return list(self.collector.rooms.keys())

    def get_instance_names_in_room(self, in_room: str) -> list[str]:
        """Get instance names that are spatially inside a given room.

        Buffers the room polygon slightly to handle mesh-sampling precision.

        Args:
            in_room: Exact room key (must exist in get_room_names()).

        Returns:
            List of instance names within the room.

        """
        room_poly = self.collector.rooms.get(in_room)
        if room_poly is None:
            return self.get_instance_names()
        room_buffered = room_poly.buffer(0.1)
        all_names = self.get_instance_names()
        return [
            k
            for k in all_names
            if room_buffered.contains(
                self.collector.instances[k].representative_point()
            )
        ]

    def resolve_on_instance(
        self,
        on_instance: str,
        gpt_client: object | None = None,
    ) -> str | None:
        r"""Resolve on_instance to an exact key (for placement).

        If on_instance is already in get_instance_names(), return it.
        Otherwise if gpt_client is provided, use LLM to resolve user description
        (e.g. \"柜子\", \"书柜\") to one exact instance key.

        Args:
            on_instance: Exact instance key or semantic description.
            gpt_client: Optional GPT client for semantic resolve (e.g. GPT_CLIENT).

        Returns:
            Exact instance key, or None if not found / LLM returned NONE.
        """
        names = self.get_instance_names()
        if on_instance in names:
            return on_instance
        if gpt_client is not None:
            return resolve_instance_with_llm(
                gpt_client, names, on_instance  # type: ignore[arg-type]
            )
        return None

    def resolve_in_room(
        self,
        in_room: str,
        gpt_client: object | None = None,
    ) -> str | None:
        r"""Resolve in_room to an exact room name (for placement).

        If in_room is already in get_room_names(), return it.
        Otherwise if gpt_client is provided, use LLM to resolve user description
        (e.g. \"kitchen\", \"the place for cooking\") to one exact room name.

        Args:
            in_room: Exact room name or semantic description.
            gpt_client: Optional GPT client for semantic resolve (e.g. GPT_CLIENT).

        Returns:
            Exact room name, or None if not found / LLM returned NONE.
        """
        names = self.get_room_names()
        if in_room in names:
            return in_room
        if gpt_client is not None:
            return resolve_instance_with_llm(
                gpt_client, names, in_room  # type: ignore[arg-type]
            )
        return None

    def resolve_beside_instance(
        self,
        beside_instance: str,
        gpt_client: object | None = None,
        in_room: str | None = None,
    ) -> str | None:
        r"""Resolve beside_instance to an exact key (for beside placement).

        If beside_instance is already in get_instance_names(), return it.
        Otherwise if gpt_client is provided, use LLM to resolve user description
        (e.g. "桌子", "沙发") to one exact instance key.

        When `in_room` is given, only instances spatially inside that room are
        considered as candidates.

        Args:
            beside_instance: Exact instance key or semantic description.
            gpt_client: Optional GPT client for semantic resolve.
            in_room: Optional resolved room key to restrict candidate scope.

        Returns:
            Exact instance key, or None if not found / LLM returned NONE.
        """
        if in_room is not None:
            names = self.get_instance_names_in_room(in_room)
        else:
            names = self.get_instance_names()
        if beside_instance in names:
            return beside_instance

        # Substring matching as fallback
        query_lower = beside_instance.lower()
        matches = [n for n in names if query_lower in n.lower()]
        if len(matches) == 1:
            logger.info(
                "Substring match: '%s' -> '%s'", beside_instance, matches[0]
            )
            return matches[0]
        elif len(matches) > 1:
            logger.warning(
                "Multiple substring matches for '%s': %s. Using '%s'. "
                "Use exact name or LLM for better matching.",
                beside_instance,
                matches,
                matches[0],
            )
            return matches[0]

        if gpt_client is not None:
            return resolve_instance_with_llm(
                gpt_client, names, beside_instance  # type: ignore[arg-type]
            )
        return None

    def resolve_delete_instance(
        self,
        delete_instance: str,
        gpt_client: object | None = None,
        in_room: str | None = None,
    ) -> str | None:
        r"""Resolve delete_instance to an exact key (for deletion).

        Similar to resolve_beside_instance but specifically for deletion.
        If delete_instance is already in get_instance_names(), return it.
        Otherwise if gpt_client is provided, use LLM to resolve user description
        (e.g. "桌子", "沙发") to one exact instance key.

        When `in_room` is given, only instances spatially inside that room are
        considered as candidates.

        Args:
            delete_instance: Exact instance key or semantic description.
            gpt_client: Optional GPT client for semantic resolve.
            in_room: Optional resolved room key to restrict candidate scope.

        Returns:
            Exact instance key, or None if not found / LLM returned NONE.
        """
        if in_room is not None:
            names = self.get_instance_names_in_room(in_room)
        else:
            names = self.get_instance_names()

        if delete_instance in names:
            return delete_instance

        # Substring matching as fallback
        query_lower = delete_instance.lower()
        matches = [n for n in names if query_lower in n.lower()]
        if len(matches) == 1:
            logger.info(
                "Substring match: '%s' -> '%s'", delete_instance, matches[0]
            )
            return matches[0]
        elif len(matches) > 1:
            logger.warning(
                "Multiple substring matches for '%s': %s. Using '%s'. "
                "Use exact name or LLM for better matching.",
                delete_instance,
                matches,
                matches[0],
            )
            return matches[0]

        if gpt_client is not None:
            return resolve_instance_with_llm(
                gpt_client, names, delete_instance  # type: ignore[arg-type]
            )
        return None

    def query_instance_center(
        self,
        instance_key: str,
    ) -> list[float] | None:
        """Query the center coordinates of an instance.

        Args:
            instance_key: Exact instance name to query.

        Returns:
            List [x, y, z] of the instance center, or None if not found.

        """
        return self.collector.get_instance_center(instance_key)

    def resolve_and_query_instance(
        self,
        query_instance: str,
        gpt_client: object | None = None,
    ) -> tuple[str | None, list[float] | None]:
        r"""Resolve instance name and return its center coordinates.

        Combines fuzzy matching with coordinate query.
        If query_instance is already in get_instance_names(), return its center.
        Otherwise if gpt_client is provided, use LLM to resolve user description
        (e.g. "床", "沙发") to one exact instance key.

        Args:
            query_instance: Exact instance key or semantic description.
            gpt_client: Optional GPT client for semantic resolve.

        Returns:
            Tuple of (resolved_instance_name, center_coordinates), or (None, None) if not found.

        """
        names = self.get_instance_names()

        if query_instance in names:
            center = self.query_instance_center(query_instance)
            return query_instance, center

        # Substring matching as fallback
        query_lower = query_instance.lower()
        matches = [n for n in names if query_lower in n.lower()]
        if len(matches) == 1:
            logger.info(
                "Substring match: '%s' -> '%s'", query_instance, matches[0]
            )
            center = self.query_instance_center(matches[0])
            return matches[0], center
        elif len(matches) > 1:
            logger.warning(
                "Multiple substring matches for '%s': %s. Using '%s'. "
                "Use exact name or LLM for better matching.",
                query_instance,
                matches,
                matches[0],
            )
            center = self.query_instance_center(matches[0])
            return matches[0], center

        if gpt_client is not None:
            resolved = resolve_instance_with_llm(
                gpt_client, names, query_instance  # type: ignore[arg-type]
            )
            if resolved:
                center = self.query_instance_center(resolved)
                return resolved, center

        return None, None

    def get_occupied_area(self) -> Geometry:
        """Get the union of all occupied areas.

        Returns:
            Shapely geometry representing occupied areas.

        """
        return self.collector.occ_area

    def get_floor_union(self) -> Geometry:
        """Get the union of all floor areas.

        Returns:
            Shapely geometry representing floor areas.

        """
        return self.collector.floor_union


def visualize_floorplan(
    urdf_path: str,
    output_path: str,
    mesh_sample_num: int = DEFAULT_MESH_SAMPLE_NUM,
    ignore_items: list[str] | None = None,
) -> None:
    """Quick function to visualize a floorplan.

    Args:
        urdf_path: Path to the URDF file.
        output_path: Path to save the output image.
        mesh_sample_num: Number of points to sample from meshes.
        ignore_items: List of item name patterns to ignore.

    """
    manager = FloorplanManager(
        urdf_path=urdf_path,
        mesh_sample_num=mesh_sample_num,
        ignore_items=ignore_items,
    )
    manager.visualize(output_path=output_path)


def insert_object_to_scene(
    urdf_path: str,
    asset_path: str,
    instance_key: str,
    output_path: str,
    usd_path: str | None = None,
    in_room: str | None = None,
    on_instance: str | None = None,
    beside_instance: str | None = None,
    beside_distance: float = DEFAULT_BESIDE_DISTANCE,
    place_strategy: Literal["top", "random"] = "random",
    rotation_rpy: tuple[float, float, float] = DEFAULT_ROTATION_RPY,
) -> list[float] | None:
    """Quick function to insert an object and generate floorplan.

    Note: USD updates require Blender (bpy) to convert .obj to .usdc format.

    Args:
        urdf_path: Path to the URDF file.
        asset_path: Path to the asset mesh file (.obj).
        instance_key: Unique key for the new instance.
        output_path: Path to save the floorplan image.
        usd_path: Optional path to the USD file (requires Blender).
        in_room: Optional room name to constrain placement.
        on_instance: Optional instance name to place on top of.
        beside_instance: Optional instance name to place beside (on floor).
        beside_distance: Max distance for beside placement (meters).
        place_strategy: Either "top" or "random".
        rotation_rpy: Initial rotation in roll-pitch-yaw.

    Returns:
        List [x, y, z] of the placed instance center, or None if failed.

    """
    manager = FloorplanManager(urdf_path=urdf_path, usd_path=usd_path)
    center = manager.insert_object(
        asset_path=asset_path,
        instance_key=instance_key,
        in_room=in_room,
        on_instance=on_instance,
        beside_instance=beside_instance,
        beside_distance=beside_distance,
        rotation_rpy=rotation_rpy,
        place_strategy=place_strategy,
    )
    if center is not None:
        manager.visualize(output_path=output_path)
    return center


def delete_object_from_scene(
    urdf_path: str,
    instance_key: str,
    in_room: str | None = None,
    output_path: str | None = None,
) -> bool:
    """Quick function to delete an object from scene.

    Args:
        urdf_path: Path to the URDF file.
        instance_key: Exact instance name to delete.
        in_room: Optional room constraint - only delete if instance is in this room.
        output_path: Optional path to save the floorplan image after deletion.

    Returns:
        True if deletion succeeded, False otherwise.

    """
    manager = FloorplanManager(urdf_path=urdf_path)
    success = manager.delete_object(
        instance_key=instance_key,
        in_room=in_room,
    )
    if success and output_path is not None:
        manager.visualize(output_path=output_path)
    return success


def query_instance_position(
    urdf_path: str,
    instance_key: str,
) -> list[float] | None:
    """Quick function to query instance center coordinates.

    Args:
        urdf_path: Path to the URDF file.
        instance_key: Exact instance name to query.

    Returns:
        List [x, y, z] of the instance center, or None if not found.

    """
    manager = FloorplanManager(urdf_path=urdf_path)
    return manager.query_instance_center(instance_key)
