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
from dataclasses import dataclass

import tyro
from embodied_gen.scripts.simulate_sapien import entrypoint as sim_cli
from embodied_gen.utils.enum import LayoutInfo
from embodied_gen.utils.geometry import bfs_placement, compose_mesh_scene
from embodied_gen.utils.log import logger


@dataclass
class LayoutPlacementConfig:
    layout_path: str
    output_dir: str | None = None
    seed: int | None = None
    max_attempts: int = 1000
    output_iscene: bool = False
    insert_robot: bool = False


def entrypoint(**kwargs):
    if kwargs is None or len(kwargs) == 0:
        args = tyro.cli(LayoutPlacementConfig)
    else:
        args = LayoutPlacementConfig(**kwargs)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.dirname(args.layout_path)
    )
    os.makedirs(output_dir, exist_ok=True)
    out_scene_path = f"{output_dir}/Iscene.glb"
    out_layout_path = f"{output_dir}/layout.json"

    layout_info = bfs_placement(args.layout_path, seed=args.seed)
    with open(out_layout_path, "w") as f:
        json.dump(layout_info.to_dict(), f, indent=4)

    if args.output_iscene:
        compose_mesh_scene(layout_info, out_scene_path)

    sim_cli(
        layout_path=out_layout_path,
        output_dir=output_dir,
        robot_name="franka" if args.insert_robot else None,
    )

    logger.info(f"Layout placement completed in {output_dir}")


if __name__ == "__main__":
    entrypoint()
