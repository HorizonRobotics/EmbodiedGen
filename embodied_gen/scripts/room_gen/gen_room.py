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


import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum

import tyro
from embodied_gen.utils.log import logger

EXEC_PYTHON = os.environ.get(
    "BLENDER_PYTHON",
    "thirdparty/infinigen/blender/4.2/python/bin/python3.11",
)


class RoomType(str, Enum):
    bedroom = "Bedroom"
    livingRoom = "LivingRoom"
    kitchen = "Kitchen"
    bathroom = "Bathroom"
    diningRoom = "DiningRoom"
    office = "Office"
    house = "House"


class Complexity(str, Enum):
    minimalist = "minimalist"
    simple = "simple"
    medium = "medium"
    detail = "detail"


@dataclass
class GenRoomArgs:
    """Configuration for single-seed Infinigen room generation and export."""

    output_root: str
    """The base output directory for generated rooms."""

    room_type: RoomType = RoomType.kitchen
    """The type of room to generate."""

    seed: int = None
    """The specific seed number to generate."""

    # Task Switches (Default to True, use flags like --no-gen to disable)
    gen: bool = True
    """Whether to run the indoor generation task (generate_indoors)."""

    urdf: bool = True
    """Whether to export to URDF (requires generation output)."""

    usd: bool = True
    """Whether to export to USD (requires generation output)."""

    custom_params: str = "embodied_gen/scripts/room_gen/custom_solve.gin"

    large_scene: bool = False
    """If True, has_fewer_rooms=False for large scene generation."""

    complexity: Complexity = Complexity.simple
    """Complexity level: minimalist, simple, medium, or detail."""

    prompt: str | None = None
    """Natural-language task/scene description. When set, room_type and
    complexity are inferred from it via a GPT router, overriding the
    explicit values."""


def run_command(cmd: list[str], task_name: str):
    """Helper: Use Popen to allow killing the child process on Ctrl+C.

    Includes execution time logging.
    """
    logger.info(f"--> Running {task_name}...")
    start_time = time.time()
    process = subprocess.Popen(cmd, env=None)
    try:
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        elapsed_mins = (time.time() - start_time) / 60
        logger.info(
            f"--> {task_name} successfully in {elapsed_mins:.1f} mins."
        )

    except KeyboardInterrupt:
        logger.warning(f"\n[Interrupted] Stopping {task_name}...")
        process.kill()
        process.wait()
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        logger.info(
            f"Error occurred during {task_name}. Exit code: {e.returncode}"
        )
        sys.exit(1)


def generate_room(cfg: GenRoomArgs):
    if cfg.prompt:
        # Deferred import: GPT routing deps are only needed in prompt mode.
        from embodied_gen.scripts.room_gen.route_room import (
            InfinigenGenRouter,
        )
        from embodied_gen.utils.gpt_clients import GPT_CLIENT

        router = InfinigenGenRouter(gpt_client=GPT_CLIENT)
        room_name, complexity = router.query(cfg.prompt)
        cfg.room_type = RoomType(room_name)
        cfg.complexity = Complexity(complexity)
        logger.info(
            f"Router mapped prompt {cfg.prompt!r} to room={room_name}, "
            f"complexity={complexity}."
        )

    room_type = cfg.room_type.value
    seed = cfg.seed
    if seed is None:
        seed = random.randint(0, 100000)

    blender_dir = f"{cfg.output_root}/{room_type}_seed{seed}/blender"
    logger.info(
        f"{room_type} | Seed {seed}: Gen={cfg.gen}, URDF={cfg.urdf}, USD={cfg.usd}"
    )

    # Complexity configuration mapping
    complexity_config = {
        Complexity.minimalist: {
            "compose_indoors.solve_large_enabled": False,
            "compose_indoors.solve_medium_enabled": False,
            "compose_indoors.solve_small_enabled": False,
        },
        Complexity.simple: {
            "compose_indoors.solve_large_enabled": True,
            "compose_indoors.solve_medium_enabled": False,
            "compose_indoors.solve_small_enabled": False,
        },
        Complexity.medium: {
            "compose_indoors.solve_large_enabled": True,
            "compose_indoors.solve_medium_enabled": True,
            "compose_indoors.solve_small_enabled": False,
        },
        Complexity.detail: {
            "compose_indoors.solve_large_enabled": True,
            "compose_indoors.solve_medium_enabled": True,
            "compose_indoors.solve_small_enabled": True,
        },
    }

    # Get complexity settings
    complexity_settings = complexity_config[cfg.complexity]
    time_cost_info = {
        Complexity.minimalist: "~1mins",
        Complexity.simple: "~10mins",
        Complexity.medium: "~20mins",
        Complexity.detail: "~70mins",
    }
    logger.info(
        f"Complexity: {cfg.complexity.value} (estimated time: {time_cost_info[cfg.complexity]})"
    )

    if cfg.gen:
        dst_gin = "thirdparty/infinigen/infinigen_examples/configs_indoor/custom_solve.gin"
        shutil.copy(cfg.custom_params, dst_gin)
        cmd_generate = [
            EXEC_PYTHON,
            "embodied_gen/scripts/room_gen/run_generate_indoors.py",
            "--seed",
            str(seed),
            "--task",
            "coarse",
            "--output_folder",
            blender_dir,
            "-g",
            "custom_solve.gin",
        ]
        if room_type == "House":
            has_fewer_rooms_value = "False" if cfg.large_scene else "True"
            cmd_generate.append("-p")
            cmd_generate.append(
                f'home_room_constraints.has_fewer_rooms={has_fewer_rooms_value}'
            )
        else:
            cmd_generate.append("-p")
            cmd_generate.append(
                f'restrict_solving.restrict_parent_rooms=["{room_type}"]'
            )
            cmd_generate.append("restrict_solving.solve_max_rooms=1")
            if room_type == "Office":
                cmd_generate.append("home_room_constraints.office_only=True")
        cmd_generate.append(
            f"compose_indoors.solve_large_enabled={complexity_settings['compose_indoors.solve_large_enabled']}"
        )
        cmd_generate.append(
            f"compose_indoors.solve_medium_enabled={complexity_settings['compose_indoors.solve_medium_enabled']}"
        )
        cmd_generate.append(
            f"compose_indoors.solve_small_enabled={complexity_settings['compose_indoors.solve_small_enabled']}"
        )
        run_command(cmd_generate, "Room Generation")

    if cfg.urdf:
        if not os.path.exists(blender_dir) and not cfg.gen:
            logger.warning(f"Warning: {blender_dir} not found. Skipping URDF.")
        else:
            cmd_export_urdf = [
                EXEC_PYTHON,
                "embodied_gen/scripts/room_gen/export_scene.py",
                "--input_folder",
                blender_dir,
                "--output_folder",
                f"{cfg.output_root}/{room_type}_seed{seed}/urdf",
                "-f",
                "obj",
                "-r",
                "1024",
                "--individual",
                "--deconvex",
                "--center_scene",
            ]
            run_command(cmd_export_urdf, "Export URDF")

    if cfg.usd:
        if not os.path.exists(blender_dir) and not cfg.gen:
            logger.warning(f"Warning: {blender_dir} not found. Skipping USD.")
        else:
            cmd_export_usd = [
                EXEC_PYTHON,
                "embodied_gen/scripts/room_gen/export_scene.py",
                "--input_folder",
                blender_dir,
                "--output_folder",
                f"{cfg.output_root}/{room_type}_seed{seed}/usd",
                "-f",
                "usdc",
                "-r",
                "1024",
                "--omniverse",
                "--center_scene",
            ]
            run_command(cmd_export_usd, "Export USD")

    logger.info(f"\n=== Completed {room_type} Seed {seed} ===")


if __name__ == "__main__":
    try:
        cfg = tyro.cli(GenRoomArgs)
        generate_room(cfg)
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user (Cmd+C). Exiting.")
        sys.exit(0)
