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


import argparse
import logging
import os

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import (
    StableDiffusionXLPipeline,
)
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import (  # noqa
    StableDiffusionXLPipeline as StableDiffusionXLPipelineIP,
)
from tqdm import tqdm
from embodied_gen.models.text_model import (
    build_text2img_ip_pipeline,
    build_text2img_pipeline,
    text2img_gen,
)
from embodied_gen.utils.process_media import parse_text_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Text to Image.")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="List of prompts (space-separated).",
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        nargs="+",
        help="List of ref_image paths (space-separated).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Root directory for saving outputs.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=12.0,
        help="Guidance scale for the diffusion model.",
    )
    parser.add_argument(
        "--ref_scale",
        type=float,
        default=0.3,
        help="Reference image scale for the IP adapter.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--infer_step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    return args


def entrypoint(
    pipeline: StableDiffusionXLPipeline | StableDiffusionXLPipelineIP = None,
    **kwargs,
) -> list[str]:
    args = parse_args()
    for k, v in kwargs.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    prompts = parse_text_prompts(args.prompts)
    os.makedirs(args.output_root, exist_ok=True)

    ip_img_paths = args.ref_image
    if ip_img_paths is None or len(ip_img_paths) == 0:
        args.ref_scale = 0
        ip_img_paths = [None] * len(prompts)
    elif isinstance(ip_img_paths, str):
        ip_img_paths = [ip_img_paths] * len(prompts)
    elif isinstance(ip_img_paths, list):
        if len(ip_img_paths) == 1:
            ip_img_paths = ip_img_paths * len(prompts)
    else:
        raise ValueError("Invalid ref_image paths.")
    assert len(ip_img_paths) == len(
        prompts
    ), f"Number of ref images does not match prompts, {len(ip_img_paths)} != {len(prompts)}"  # noqa

    if pipeline is None:
        if args.ref_scale > 0:
            pipeline = build_text2img_ip_pipeline(
                "weights/Kolors",
                ref_scale=args.ref_scale,
            )
        else:
            pipeline = build_text2img_pipeline("weights/Kolors")

    for idx, (prompt, ip_img_path) in tqdm(
        enumerate(zip(prompts, ip_img_paths)),
        desc="Generating images",
        total=len(prompts),
    ):
        images = text2img_gen(
            prompt=prompt,
            n_sample=args.n_sample,
            guidance_scale=args.guidance_scale,
            pipeline=pipeline,
            ip_image=ip_img_path,
            image_wh=[args.resolution, args.resolution],
            infer_step=args.infer_step,
            seed=args.seed,
        )

        save_paths = []
        for sub_idx, image in enumerate(images):
            save_path = (
                f"{args.output_root}/sample_{idx*args.n_sample+sub_idx}.png"
            )
            image.save(save_path)
            save_paths.append(save_path)

        logger.info(f"Images saved to {args.output_root}")

    return save_paths


if __name__ == "__main__":
    entrypoint()
