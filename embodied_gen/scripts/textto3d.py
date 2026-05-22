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
import types
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from embodied_gen.models.image_comm_model import build_hf_image_pipeline
from embodied_gen.models.segment_model import RembgRemover
from embodied_gen.models.text_model import PROMPT_APPEND
from embodied_gen.scripts.imageto3d import (
    IMAGE3D_MODEL,
    SUPPORTED_IMAGE3D_MODELS,
)
from embodied_gen.scripts.imageto3d import entrypoint as imageto3d_api
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import (
    check_object_edge_truncated,
    combine_images_to_grid,
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

# TXTGEN_CHECKER drives the final text↔3D quality gate for every backend
# (SAM3D / TRELLIS / HUNYUAN3D), so it stays eager.
TXTGEN_CHECKER = TextGenAlignChecker(GPT_CLIENT)

# The text-to-image stack (PIPE_IMG, BG_REMOVER, SEMANTIC_CHECKER, SEG_CHECKER)
# is only used on the SAM3D / TRELLIS path. HUNYUAN3D goes directly from
# prompt to 3D, so we lazily construct these at first use to avoid loading
# the SD/Flux pipeline (and downstream checkers) when not needed.
SEMANTIC_CHECKER = None
SEG_CHECKER = None
PIPE_IMG = None
BG_REMOVER = None


def _ensure_text2img_stack() -> None:
    """Construct the text-to-image pipeline + image-stage checkers once.

    Called from the SAM3D / TRELLIS path before any ``text_to_image`` run.
    Idempotent: subsequent calls return immediately.
    """
    global SEMANTIC_CHECKER, SEG_CHECKER, PIPE_IMG, BG_REMOVER
    if PIPE_IMG is not None:
        return
    logger.info("Loading TEXT2IMG_MODEL...")
    SEMANTIC_CHECKER = SemanticConsistChecker(GPT_CLIENT)
    SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
    PIPE_IMG = build_hf_image_pipeline(os.environ.get("TEXT_MODEL", "sd35"))
    BG_REMOVER = RembgRemover()


__all__ = [
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
    _ensure_text2img_stack()
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

    args.image3d_model = str(args.image3d_model).strip().upper()
    if args.image3d_model not in SUPPORTED_IMAGE3D_MODELS:
        raise ValueError(
            f"Unsupported --image3d_model {args.image3d_model!r}; "
            f"expected one of {SUPPORTED_IMAGE3D_MODELS}."
        )

    hunyuan_cfg = None
    hunyuan_creds = None
    process_prompt = None
    if args.image3d_model == "HUNYUAN3D":
        # Local import keeps SAM3D/TRELLIS callers from pulling the hunyuan
        # module (and its kaolin/PIL chain) when not selected.
        from embodied_gen.models.hunyuan3d import (
            HunyuanConfig,
            load_credentials,
            process_prompt,
        )

        # Fail fast on missing creds before any network I/O.
        hunyuan_creds = load_credentials()
        hunyuan_cfg = HunyuanConfig()
        logger.info(
            "HUNYUAN3D text-to-3D backend: action=%s host=%s result_format=%s",
            hunyuan_cfg.image_action,
            hunyuan_cfg.host,
            hunyuan_cfg.result_format,
        )

    if args.asset_names is None or len(args.asset_names) == 0:
        args.asset_names = [f"sample3d_{i}" for i in range(len(args.prompts))]
    asset_save_dir = os.path.join(args.output_root, "asset3d")
    os.makedirs(asset_save_dir, exist_ok=True)
    # HUNYUAN3D path skips text-to-image entirely; the images/ dir only
    # exists when the local SAM3D / TRELLIS pipeline produces conditioning
    # images.
    img_save_dir = os.path.join(args.output_root, "images")
    if args.image3d_model != "HUNYUAN3D":
        os.makedirs(img_save_dir, exist_ok=True)
    results = defaultdict(dict)
    for idx, (prompt, node) in enumerate(zip(args.prompts, args.asset_names)):
        success_flag = False
        n_pipe_retry = args.n_pipe_retry
        seed_img = args.seed_img
        seed_3d = args.seed_3d
        # Tencent Pro API is charged per submit; force a single attempt to
        # avoid silently multiplying cost when --n_pipe_retry > 1.
        if args.image3d_model == "HUNYUAN3D" and n_pipe_retry > 1:
            logger.warning(
                "HUNYUAN3D mode: --n_pipe_retry forced to 1 (Tencent API "
                "is charged per submit); user passed %d.",
                n_pipe_retry,
            )
            n_pipe_retry = 1
        while success_flag is False and n_pipe_retry > 0:
            logger.info(
                f"GEN pipeline for node {node}\n"
                f"Try round: {args.n_pipe_retry-n_pipe_retry+1}/{args.n_pipe_retry}, Prompt: {prompt}"
            )
            save_node = node.replace(" ", "_")
            node_save_dir = f"{asset_save_dir}/{save_node}"
            asset_type = node if "sample3d_" not in node else None

            if args.image3d_model == "HUNYUAN3D":
                # Skip text-to-image + SEMANTIC/SEG/AESTHETIC checkers;
                # drive Hunyuan3D Pro text-to-3D directly so the result
                # tree lands at <node_save_dir>/result/ matching the
                # SAM3D / TRELLIS layout.
                hunyuan_args = types.SimpleNamespace(
                    asset_type=[asset_type],
                    version=None,
                    height_range=None,
                    mass_range=None,
                    disable_decompose_convex=args.disable_decompose_convex,
                    keep_intermediate=args.keep_intermediate,
                )
                process_prompt(
                    args=hunyuan_args,
                    idx=0,
                    prompt=prompt,
                    output_root=node_save_dir,
                    filename=save_node,
                    hunyuan_config=hunyuan_cfg,
                    hunyuan_credentials=hunyuan_creds,
                    checkers=[],
                )
            else:
                # Text-to-image GEN (SAM3D / TRELLIS path).
                gen_image_path = f"{img_save_dir}/{save_node}.png"
                text_to_image(
                    prompt,
                    gen_image_path,
                    args.n_image_retry,
                    args.img_denoise_step,
                    args.text_guidance_scale,
                    args.n_img_sample,
                    seed=seed_img,
                )

                # Asset 3D GEN
                imageto3d_api(
                    image_path=[gen_image_path],
                    output_root=node_save_dir,
                    asset_type=[asset_type],
                    seed=(
                        random.randint(0, 100000)
                        if seed_3d is None
                        else seed_3d
                    ),
                    n_retry=args.n_asset_retry,
                    keep_intermediate=args.keep_intermediate,
                    disable_decompose_convex=args.disable_decompose_convex,
                    image3d_model=args.image3d_model,
                )
            mesh_path = f"{node_save_dir}/result/mesh/{save_node}.obj"
            image_path = render_asset3d(
                mesh_path,
                output_root=f"{node_save_dir}/result",
                num_images=4,
                elevation=(30, -30),
                output_subdir="renders",
                no_index_file=True,
            )
            image_path = combine_images_to_grid(image_path)
            check_text = asset_type if asset_type is not None else prompt
            qa_flag, qa_result = TXTGEN_CHECKER(check_text, image_path)
            logger.warning(
                f"Node {node}, {TXTGEN_CHECKER.__class__.__name__}: {qa_result}"
            )
            results["assets"][node] = f"asset3d/{save_node}/result"
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
    parser.add_argument(
        "--image3d_model",
        type=str,
        default=IMAGE3D_MODEL,
        help=(
            "Image-to-3D backend selector forwarded to imageto3d. One of "
            f"{', '.join(SUPPORTED_IMAGE3D_MODELS)} (case-insensitive). "
            "HUNYUAN3D skips the text-to-image stage entirely and calls "
            "Tencent Hunyuan3D Pro text-to-3D directly; it requires "
            "TENCENT_SECRET_ID/TENCENT_SECRET_KEY in the environment."
        ),
    )

    args, unknown = parser.parse_known_args()

    return args


if __name__ == "__main__":
    text_to_3d()
