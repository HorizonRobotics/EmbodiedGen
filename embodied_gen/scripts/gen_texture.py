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
import shutil
from dataclasses import dataclass

import tyro
from embodied_gen.data.backproject_v2 import entrypoint as backproject_api
from embodied_gen.data.differentiable_render import entrypoint as drender_api
from embodied_gen.data.utils import as_list
from embodied_gen.models.delight_model import DelightingModel
from embodied_gen.models.sr_model import ImageRealESRGAN
from embodied_gen.scripts.render_mv import (
    build_texture_gen_pipe,
)
from embodied_gen.scripts.render_mv import infer_pipe as render_mv_api
from embodied_gen.utils.log import logger


@dataclass
class TextureGenConfig:
    mesh_path: str | list[str]
    prompt: str | list[str]
    output_root: str
    controlnet_cond_scale: float = 0.7
    guidance_scale: float = 9
    strength: float = 0.9
    num_inference_steps: int = 40
    delight: bool = True
    seed: int = 0
    base_ckpt_dir: str = "./weights"
    texture_size: int = 2048
    ip_adapt_scale: float = 0.0
    ip_img_path: str | list[str] | None = None


def entrypoint() -> None:
    cfg = tyro.cli(TextureGenConfig)
    cfg.mesh_path = as_list(cfg.mesh_path)
    cfg.prompt = as_list(cfg.prompt)
    cfg.ip_img_path = as_list(cfg.ip_img_path)
    assert len(cfg.mesh_path) == len(cfg.prompt)

    # Pre-load models.
    if cfg.ip_adapt_scale > 0:
        PIPELINE = build_texture_gen_pipe(
            base_ckpt_dir="./weights",
            ip_adapt_scale=cfg.ip_adapt_scale,
            device="cuda",
        )
    else:
        PIPELINE = build_texture_gen_pipe(
            base_ckpt_dir="./weights",
            ip_adapt_scale=0,
            device="cuda",
        )
    DELIGHT = None
    if cfg.delight:
        DELIGHT = DelightingModel()
    IMAGESR_MODEL = ImageRealESRGAN(outscale=4)

    for idx in range(len(cfg.mesh_path)):
        mesh_path = cfg.mesh_path[idx]
        prompt = cfg.prompt[idx]
        uuid = os.path.splitext(os.path.basename(mesh_path))[0]
        output_root = os.path.join(cfg.output_root, uuid)
        drender_api(
            mesh_path=mesh_path,
            output_root=f"{output_root}/condition",
            uuid=uuid,
        )
        render_mv_api(
            index_file=f"{output_root}/condition/index.json",
            controlnet_cond_scale=cfg.controlnet_cond_scale,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.strength,
            num_inference_steps=cfg.num_inference_steps,
            ip_adapt_scale=cfg.ip_adapt_scale,
            ip_img_path=(
                None if cfg.ip_img_path is None else cfg.ip_img_path[idx]
            ),
            prompt=prompt,
            save_dir=f"{output_root}/multi_view",
            sub_idxs=[[0, 1, 2], [3, 4, 5]],
            pipeline=PIPELINE,
            seed=cfg.seed,
        )
        textured_mesh = backproject_api(
            delight_model=DELIGHT,
            imagesr_model=IMAGESR_MODEL,
            mesh_path=mesh_path,
            color_path=f"{output_root}/multi_view/color_sample0.png",
            output_path=f"{output_root}/texture_mesh/{uuid}.obj",
            save_glb_path=f"{output_root}/texture_mesh/{uuid}.glb",
            skip_fix_mesh=True,
            delight=cfg.delight,
            no_save_delight_img=True,
            texture_wh=[cfg.texture_size, cfg.texture_size],
            no_mesh_post_process=True,
        )
        drender_api(
            mesh_path=f"{output_root}/texture_mesh/{uuid}.obj",
            output_root=f"{output_root}/texture_mesh",
            uuid=uuid,
            num_images=90,
            elevation=[20],
            with_mtl=True,
            gen_color_mp4=True,
            pbr_light_factor=1.2,
        )

        # Re-organize folders
        shutil.rmtree(f"{output_root}/condition")
        shutil.copy(
            f"{output_root}/texture_mesh/{uuid}/color.mp4",
            f"{output_root}/color.mp4",
        )
        shutil.rmtree(f"{output_root}/texture_mesh/{uuid}")

        logger.info(
            f"Successfully generate textured mesh in {output_root}/texture_mesh"
        )


if __name__ == "__main__":
    entrypoint()
