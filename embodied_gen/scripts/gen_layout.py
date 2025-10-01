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

import gc
import json
import os
from dataclasses import dataclass, field
from shutil import copytree
from time import time
from typing import Optional

import torch
import tyro
from embodied_gen.models.layout import build_scene_layout
from embodied_gen.scripts.simulate_sapien import entrypoint as sim_cli
from embodied_gen.scripts.textto3d import text_to_3d
from embodied_gen.utils.config import GptParamsConfig
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.geometry import bfs_placement, compose_mesh_scene
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import (
    load_scene_dict,
    parse_text_prompts,
)
from embodied_gen.validators.quality_checkers import SemanticMatcher

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class LayoutGenConfig:
    task_descs: list[str]
    output_root: str
    bg_list: str = "outputs/bg_scenes/scene_list.txt"
    n_img_sample: int = 3
    text_guidance_scale: float = 7.0
    img_denoise_step: int = 25
    n_image_retry: int = 4
    n_asset_retry: int = 3
    n_pipe_retry: int = 2
    seed_img: Optional[int] = None
    seed_3d: Optional[int] = None
    seed_layout: Optional[int] = None
    keep_intermediate: bool = False
    output_iscene: bool = False
    insert_robot: bool = False
    gpt_params: GptParamsConfig = field(
        default_factory=lambda: GptParamsConfig(
            temperature=1.0,
            top_p=0.95,
            frequency_penalty=0.3,
            presence_penalty=0.5,
        )
    )


def entrypoint() -> None:
    args = tyro.cli(LayoutGenConfig)
    SCENE_MATCHER = SemanticMatcher(GPT_CLIENT)
    task_descs = parse_text_prompts(args.task_descs)
    scene_dict = load_scene_dict(args.bg_list)
    gpt_params = args.gpt_params.to_dict()
    for idx, task_desc in enumerate(task_descs):
        logger.info(f"Generate Layout and 3D scene for task: {task_desc}")
        output_root = f"{args.output_root}/task_{idx:04d}"
        scene_graph_path = f"{output_root}/scene_tree.jpg"
        start_time = time()
        layout_info: LayoutInfo = build_scene_layout(
            task_desc, scene_graph_path, gpt_params
        )
        prompts_mapping = {v: k for k, v in layout_info.objs_desc.items()}
        prompts = [
            v
            for k, v in layout_info.objs_desc.items()
            if layout_info.objs_mapping[k] != Scene3DItemEnum.BACKGROUND.value
        ]

        for prompt in prompts:
            node = prompts_mapping[prompt]
            generation_log = text_to_3d(
                prompts=[
                    prompt,
                ],
                output_root=output_root,
                asset_names=[
                    node,
                ],
                n_img_sample=args.n_img_sample,
                text_guidance_scale=args.text_guidance_scale,
                img_denoise_step=args.img_denoise_step,
                n_image_retry=args.n_image_retry,
                n_asset_retry=args.n_asset_retry,
                n_pipe_retry=args.n_pipe_retry,
                seed_img=args.seed_img,
                seed_3d=args.seed_3d,
                keep_intermediate=args.keep_intermediate,
            )
            layout_info.assets.update(generation_log["assets"])
            layout_info.quality.update(generation_log["quality"])

        # Background GEN (for efficiency, temp use retrieval instead)
        bg_node = layout_info.relation[Scene3DItemEnum.BACKGROUND.value]
        text = layout_info.objs_desc[bg_node]
        match_key = SCENE_MATCHER.query(
            text, str(scene_dict), params=gpt_params
        )
        n_max_attempt = 10
        while match_key not in scene_dict and n_max_attempt > 0:
            logger.error(
                f"Cannot find matched scene {match_key}, retrying left {n_max_attempt}..."
            )
            match_key = SCENE_MATCHER.query(
                text, str(scene_dict), params=gpt_params
            )
            n_max_attempt -= 1

        match_scene_path = f"{os.path.dirname(args.bg_list)}/{match_key}"
        bg_save_dir = os.path.join(output_root, "background")
        copytree(match_scene_path, bg_save_dir, dirs_exist_ok=True)
        layout_info.assets[bg_node] = "background"

        # BFS layout placement.
        layout_path = f"{output_root}/layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout_info.to_dict(), f, indent=4)

        layout_info = bfs_placement(
            layout_path,
            seed=args.seed_layout,
        )
        layout_path = f"{output_root}/layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout_info.to_dict(), f, indent=4)

        if args.output_iscene:
            compose_mesh_scene(layout_info, f"{output_root}/Iscene.glb")

        sim_cli(
            layout_path=layout_path,
            output_dir=output_root,
            insert_robot=args.insert_robot,
        )

        torch.cuda.empty_cache()
        gc.collect()

        elapsed_time = (time() - start_time) / 60
        logger.info(
            f"Layout generation done for {scene_graph_path}, layout result "
            f"in {layout_path}, finished in {elapsed_time:.2f} mins."
        )

    logger.info(f"All tasks completed in {args.output_root}")


if __name__ == "__main__":
    entrypoint()
