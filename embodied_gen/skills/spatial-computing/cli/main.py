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
import logging
import sys
import warnings

import tyro

from ..api.floorplan_api import (
    FloorplanConfig,
    FloorplanManager,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def _get_gpt_client() -> object | None:
    """Lazy-import GPT_CLIENT for semantic --on_instance resolution."""
    try:
        from embodied_gen.utils.gpt_clients import GPT_CLIENT

        return GPT_CLIENT
    except Exception:
        return None


def entrypoint(cfg: FloorplanConfig) -> None:
    """Main entry point for floorplan visualization and scene manipulation.

    Args:
        cfg: Configuration object with all parameters.

    """
    manager = FloorplanManager(
        urdf_path=cfg.urdf_path,
        usd_path=cfg.usd_path,
        mesh_sample_num=cfg.mesh_sample_num,
        ignore_items=cfg.ignore_items,
        output_strategy=cfg.output_strategy,
    )

    # List instances/rooms and exit if requested
    if cfg.list_instances:
        names = manager.get_instance_names()
        rooms = manager.get_room_names()
        logger.info(f"instance_names: {names}")
        logger.info(f"room_names: {rooms}")
        return

    # Batch insertion
    if cfg.batch_insert_config is not None:
        logger.info(
            f"Loading batch insert config from {cfg.batch_insert_config}"
        )
        with open(cfg.batch_insert_config, 'r') as f:
            objects = json.load(f)

        logger.info(f"Batch inserting {len(objects)} objects...")
        centers = manager.batch_insert_objects(objects)

        success_count = len([c for c in centers if c is not None])
        logger.info(
            f"✅ Successfully placed {success_count}/{len(objects)} objects"
        )

        if success_count < len(objects):
            logger.warning(
                f"⚠️  Failed to place {len(objects) - success_count} objects"
            )
            sys.exit(1)

        # Generate floorplan visualization if requested
        if cfg.output_path is not None:
            manager.visualize(output_path=cfg.output_path)

        return

    gpt_client = _get_gpt_client()
    on_instance = cfg.on_instance
    if on_instance is not None:
        resolved = manager.resolve_on_instance(
            on_instance, gpt_client=gpt_client
        )
        if resolved is None:
            logger.error(
                "No object matched \"%s\"。Current scene instance name: %s。",
                on_instance,
                manager.get_instance_names(),
            )
            sys.exit(1)
        on_instance = resolved
        if resolved != cfg.on_instance:
            logger.info("\"%s\" -> \"%s\"", cfg.on_instance, resolved)

    in_room = cfg.in_room
    if in_room is not None:
        resolved = manager.resolve_in_room(in_room, gpt_client=gpt_client)
        if resolved is None:
            logger.error(
                "No room matched \"%s\"。Current scene room names: %s。",
                in_room,
                manager.get_room_names(),
            )
            sys.exit(1)
        in_room = resolved
        if resolved != cfg.in_room:
            logger.info("\"%s\" -> \"%s\"", cfg.in_room, resolved)

    beside_instance = cfg.beside_instance
    if beside_instance is not None:
        resolved = manager.resolve_beside_instance(
            beside_instance, gpt_client=gpt_client, in_room=in_room
        )
        if resolved is None:
            candidates = (
                manager.get_instance_names_in_room(in_room)
                if in_room
                else manager.get_instance_names()
            )
            logger.error(
                "No object matched \"%s\"。Current %sinstance name: %s。",
                beside_instance,
                f"room '{in_room}' " if in_room else "",
                candidates,
            )
            sys.exit(1)
        beside_instance = resolved
        if resolved != cfg.beside_instance:
            logger.info("\"%s\" -> \"%s\"", cfg.beside_instance, resolved)

    delete_instance = cfg.delete_instance
    delete_in_room = cfg.delete_in_room
    if delete_instance is not None:
        # Resolve room constraint if provided
        if delete_in_room is not None:
            resolved_room = manager.resolve_in_room(
                delete_in_room, gpt_client=gpt_client
            )
            if resolved_room is None:
                logger.error(
                    "No room matched \"%s\"。Current scene room names: %s。",
                    delete_in_room,
                    manager.get_room_names(),
                )
                sys.exit(1)
            delete_in_room = resolved_room
            if resolved_room != cfg.delete_in_room:
                logger.info(
                    "\"%s\" -> \"%s\"", cfg.delete_in_room, resolved_room
                )

        # Resolve delete_instance with fuzzy matching
        resolved = manager.resolve_delete_instance(
            delete_instance, gpt_client=gpt_client, in_room=delete_in_room
        )
        if resolved is None:
            candidates = (
                manager.get_instance_names_in_room(delete_in_room)
                if delete_in_room
                else manager.get_instance_names()
            )
            logger.error(
                "No object matched \"%s\"。Current %sinstance name: %s。",
                delete_instance,
                f"room '{delete_in_room}' " if delete_in_room else "",
                candidates,
            )
            sys.exit(1)
        delete_instance = resolved
        if resolved != cfg.delete_instance:
            logger.info("\"%s\" -> \"%s\"", cfg.delete_instance, resolved)

        # Perform deletion
        success = manager.delete_object(
            instance_key=delete_instance,
            in_room=delete_in_room,
        )
        if success:
            logger.info(
                f"✅ Successfully deleted '{delete_instance}' from scene."
            )
        else:
            logger.error(f"❌ Failed to delete '{delete_instance}'.")
            sys.exit(1)

    # Query instance position
    query_instance = cfg.query_instance
    if query_instance is not None:
        # Resolve instance with fuzzy matching
        resolved_name, center = manager.resolve_and_query_instance(
            query_instance, gpt_client=gpt_client
        )

        if resolved_name is None or center is None:
            logger.error(
                "No object matched \"%s\"。Current instance names: %s。",
                query_instance,
                manager.get_instance_names(),
            )
            sys.exit(1)

        if resolved_name != query_instance:
            logger.info("\"%s\" -> \"%s\"", query_instance, resolved_name)

        logger.info(
            f"📍 Instance '{resolved_name}' center: "
            f"({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})"
        )

    # Add asset instance if specified
    center = None
    if cfg.asset_path is not None:
        center = manager.insert_object(
            asset_path=cfg.asset_path,
            instance_key=cfg.instance_key,
            in_room=in_room,
            on_instance=on_instance,
            beside_instance=beside_instance,
            beside_distance=cfg.beside_distance,
            rotation_rpy=cfg.rotation_rpy,
            n_max_attempt=cfg.max_placement_attempts,
            place_strategy=cfg.place_strategy,
        )

        if center is not None:
            logger.info(
                f"Successfully placed '{cfg.instance_key}' at "
                f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )
        else:
            logger.error(
                f"❌ Failed to place '{cfg.instance_key}' in the scene."
            )
            sys.exit(1)

    # Generate floorplan visualization
    if cfg.output_path is not None:
        manager.visualize(output_path=cfg.output_path)


if __name__ == "__main__":
    config = tyro.cli(FloorplanConfig)
    entrypoint(config)
