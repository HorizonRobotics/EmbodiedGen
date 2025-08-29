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
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from embodied_gen.models.image_comm_model import build_hf_image_pipeline
from embodied_gen.models.segment_model import RembgRemover
from embodied_gen.models.text_model import PROMPT_APPEND
from embodied_gen.scripts.imageto3d import entrypoint as imageto3d_api
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import (
    check_object_edge_truncated,
    render_asset3d,
)
from embodied_gen.validators.quality_checkers import (
    ImageSegChecker,
    SemanticConsistChecker,
    TextGenAlignChecker,
)

# Avoid huggingface/tokenizers: The current process just got forked.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(0)

logger.info("Loading TEXT2IMG_MODEL...")
SEMANTIC_CHECKER = SemanticConsistChecker(GPT_CLIENT)
SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
TXTGEN_CHECKER = TextGenAlignChecker(GPT_CLIENT)
PIPE_IMG = build_hf_image_pipeline(os.environ.get("TEXT_MODEL", "sd35"))
BG_REMOVER = RembgRemover()


__all__ = [
    "text_to_image",
    "text_to_3d",
]


def text_to_image(
    prompt: str,
    save_path: str,
    n_retry: int,
    img_denoise_step: int,
    text_guidance_scale: float,
    n_img_sample: int,
    image_hw: tuple[int, int] = (1024, 1024),
    seed: int = None,
) -> bool:
    select_image = None
    success_flag = False
    assert save_path.endswith(".png"), "Image save path must end with `.png`."
    for try_idx in range(n_retry):
        if select_image is not None:
            select_image[0].save(save_path.replace(".png", "_raw.png"))
            select_image[1].save(save_path)
            break

        f_prompt = PROMPT_APPEND.format(object=prompt)
        logger.info(
            f"Image GEN for {os.path.basename(save_path)}\n"
            f"Try: {try_idx + 1}/{n_retry}, Seed: {seed}, Prompt: {f_prompt}"
        )
        torch.cuda.empty_cache()
        images = PIPE_IMG.run(
            f_prompt,
            num_inference_steps=img_denoise_step,
            guidance_scale=text_guidance_scale,
            num_images_per_prompt=n_img_sample,
            height=image_hw[0],
            width=image_hw[1],
            generator=(
                torch.Generator().manual_seed(seed)
                if seed is not None
                else None
            ),
        )

        for idx in range(len(images)):
            raw_image: Image.Image = images[idx]
            image = BG_REMOVER(raw_image)
            image.save(save_path)
            semantic_flag, semantic_result = SEMANTIC_CHECKER(
                prompt, [image.convert("RGB")]
            )
            seg_flag, seg_result = SEG_CHECKER(
                [raw_image, image.convert("RGB")]
            )
            image_mask = np.array(image)[..., -1]
            edge_flag = check_object_edge_truncated(image_mask)
            logger.warning(
                f"SEMANTIC: {semantic_result}. SEG: {seg_result}. EDGE: {edge_flag}"
            )
            if (
                (edge_flag and semantic_flag and seg_flag)
                or (edge_flag and semantic_flag is None)
                or (edge_flag and seg_flag is None)
            ):
                select_image = [raw_image, image]
                success_flag = True
                break

        seed = random.randint(0, 100000) if seed is not None else None

    return success_flag


def text_to_3d(**kwargs) -> dict:
    args = parse_args()
    for k, v in kwargs.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    if args.asset_names is None or len(args.asset_names) == 0:
        args.asset_names = [f"sample3d_{i}" for i in range(len(args.prompts))]
    img_save_dir = os.path.join(args.output_root, "images")
    asset_save_dir = os.path.join(args.output_root, "asset3d")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(asset_save_dir, exist_ok=True)
    results = defaultdict(dict)
    for prompt, node in zip(args.prompts, args.asset_names):
        success_flag = False
        n_pipe_retry = args.n_pipe_retry
        seed_img = args.seed_img
        seed_3d = args.seed_3d
        while success_flag is False and n_pipe_retry > 0:
            logger.info(
                f"GEN pipeline for node {node}\n"
                f"Try round: {args.n_pipe_retry-n_pipe_retry+1}/{args.n_pipe_retry}, Prompt: {prompt}"
            )
            # Text-to-image GEN
            save_node = node.replace(" ", "_")
            gen_image_path = f"{img_save_dir}/{save_node}.png"
            textgen_flag = text_to_image(
                prompt,
                gen_image_path,
                args.n_image_retry,
                args.img_denoise_step,
                args.text_guidance_scale,
                args.n_img_sample,
                seed=seed_img,
            )

            # Asset 3D GEN
            node_save_dir = f"{asset_save_dir}/{save_node}"
            asset_type = node if "sample3d_" not in node else None
            imageto3d_api(
                image_path=[gen_image_path],
                output_root=node_save_dir,
                asset_type=[asset_type],
                seed=random.randint(0, 100000) if seed_3d is None else seed_3d,
                n_retry=args.n_asset_retry,
                keep_intermediate=args.keep_intermediate,
                disable_decompose_convex=args.disable_decompose_convex,
            )
            mesh_path = f"{node_save_dir}/result/mesh/{save_node}.obj"
            image_path = render_asset3d(
                mesh_path,
                output_root=f"{node_save_dir}/result",
                num_images=6,
                elevation=(30, -30),
                output_subdir="renders",
                no_index_file=True,
            )

            check_text = asset_type if asset_type is not None else prompt
            qa_flag, qa_result = TXTGEN_CHECKER(check_text, image_path)
            logger.warning(
                f"Node {node}, {TXTGEN_CHECKER.__class__.__name__}: {qa_result}"
            )
            results["assets"][node] = f"{node_save_dir}/result"
            results["quality"][node] = qa_result

            if qa_flag is None or qa_flag is True:
                success_flag = True
                break

            n_pipe_retry -= 1
            seed_img = (
                random.randint(0, 100000) if seed_img is not None else None
            )
            seed_3d = (
                random.randint(0, 100000) if seed_3d is not None else None
            )

        torch.cuda.empty_cache()

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="3D Layout Generation Config")
    parser.add_argument("--prompts", nargs="+", help="text descriptions")
    parser.add_argument(
        "--output_root",
        type=str,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--asset_names",
        type=str,
        nargs="+",
        default=None,
        help="Asset names to generate",
    )
    parser.add_argument(
        "--n_img_sample",
        type=int,
        default=3,
        help="Number of image samples to generate",
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=7,
        help="Text-to-image guidance scale",
    )
    parser.add_argument(
        "--img_denoise_step",
        type=int,
        default=25,
        help="Denoising steps for image generation",
    )
    parser.add_argument(
        "--n_image_retry",
        type=int,
        default=2,
        help="Max retry count for image generation",
    )
    parser.add_argument(
        "--n_asset_retry",
        type=int,
        default=2,
        help="Max retry count for 3D generation",
    )
    parser.add_argument(
        "--n_pipe_retry",
        type=int,
        default=1,
        help="Max retry count for 3D asset generation",
    )
    parser.add_argument(
        "--seed_img",
        type=int,
        default=None,
        help="Random seed for image generation",
    )
    parser.add_argument(
        "--seed_3d",
        type=int,
        default=0,
        help="Random seed for 3D generation",
    )
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--disable_decompose_convex", action="store_true")

    args, unknown = parser.parse_known_args()

    return args


if __name__ == "__main__":
    text_to_3d()
