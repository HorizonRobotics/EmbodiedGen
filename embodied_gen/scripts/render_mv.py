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


import logging
import os
import random
from typing import List, Tuple

import fire
import numpy as np
import torch
from diffusers.utils import make_image_grid
from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import (
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from embodied_gen.data.datasets import Asset3dGenDataset
from embodied_gen.models.texture_model import build_texture_gen_pipe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_init_noise_image(image: Image.Image) -> Image.Image:
    blurred_image = image.convert("L").filter(
        ImageFilter.GaussianBlur(radius=3)
    )

    enhancer = ImageEnhance.Contrast(blurred_image)
    image_decreased_contrast = enhancer.enhance(factor=0.5)

    return image_decreased_contrast


def infer_pipe(
    index_file: str,
    controlnet_ckpt: str = None,
    uid: str = None,
    prompt: str = None,
    controlnet_cond_scale: float = 0.4,
    control_guidance_end: float = 0.9,
    strength: float = 1.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 10,
    ip_adapt_scale: float = 0,
    ip_img_path: str = None,
    sub_idxs: List[List[int]] = None,
    num_images_per_prompt: int = 3,  # increase if want similar images.
    device: str = "cuda",
    save_dir: str = "infer_vis",
    seed: int = None,
    target_hw: tuple[int, int] = (512, 512),
    pipeline: StableDiffusionXLControlNetImg2ImgPipeline = None,
) -> str:
    # sub_idxs = [[0, 1, 2], [3, 4, 5]] # None for single image.
    if sub_idxs is None:
        sub_idxs = [[random.randint(0, 5)]]  # 6 views.
        target_hw = [2 * size for size in target_hw]

    transform_list = [
        transforms.Resize(
            target_hw, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(target_hw),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    image_transform = transforms.Compose(transform_list)
    control_transform = transforms.Compose(transform_list[:-1])

    grid_hw = (target_hw[0] * len(sub_idxs), target_hw[1] * len(sub_idxs[0]))
    dataset = Asset3dGenDataset(
        index_file, target_hw=grid_hw, sub_idxs=sub_idxs
    )

    if uid is None:
        uid = random.choice(list(dataset.meta_info.keys()))
    if prompt is None:
        prompt = dataset.meta_info[uid]["capture"]
    if isinstance(prompt, List) or isinstance(prompt, Tuple):
        prompt = ", ".join(map(str, prompt))
    # prompt += "high quality, ultra-clear, high resolution, best quality, 4k"
    # prompt += "高品质,清晰,细节"
    prompt += ", high quality, high resolution, best quality"
    # prompt += ", with diffuse lighting, showing no reflections."
    logger.info(f"Inference with prompt: {prompt}")

    negative_prompt = "nsfw,阴影,低分辨率,伪影、模糊,霓虹灯,高光,镜面反射"

    control_image = dataset.fetch_sample_grid_images(
        uid,
        attrs=["image_view_normal", "image_position", "image_mask"],
        sub_idxs=sub_idxs,
        transform=control_transform,
    )

    color_image = dataset.fetch_sample_grid_images(
        uid,
        attrs=["image_color"],
        sub_idxs=sub_idxs,
        transform=image_transform,
    )

    normal_pil, position_pil, mask_pil, color_pil = dataset.visualize_item(
        control_image,
        color_image,
        save_dir=save_dir,
    )

    if pipeline is None:
        pipeline = build_texture_gen_pipe(
            base_ckpt_dir="./weights",
            controlnet_ckpt=controlnet_ckpt,
            ip_adapt_scale=ip_adapt_scale,
            device=device,
        )

    if ip_adapt_scale > 0 and ip_img_path is not None and len(ip_img_path) > 0:
        ip_image = Image.open(ip_img_path).convert("RGB")
        ip_image = ip_image.resize(target_hw[::-1])
        ip_image = [ip_image]
        pipeline.set_ip_adapter_scale([ip_adapt_scale])
    else:
        ip_image = None

    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    init_image = get_init_noise_image(normal_pil)
    # init_image = get_init_noise_image(color_pil)

    images = []
    row_num, col_num = 2, 3
    img_save_paths = []
    while len(images) < col_num:
        image = pipeline(
            prompt=prompt,
            image=init_image,
            controlnet_conditioning_scale=controlnet_cond_scale,
            control_guidance_end=control_guidance_end,
            strength=strength,
            control_image=control_image[None, ...],
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            ip_adapter_image=ip_image,
            generator=generator,
        ).images
        images.extend(image)

    grid_image = [normal_pil, position_pil, color_pil] + images[:col_num]
    # save_dir = os.path.join(save_dir, uid)
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(col_num):
        rgba_image = Image.merge("RGBA", (*images[idx].split(), mask_pil))
        img_save_path = os.path.join(save_dir, f"color_sample{idx}.png")
        rgba_image.save(img_save_path)
        img_save_paths.append(img_save_path)

    sub_idxs = "_".join(
        [str(item) for sublist in sub_idxs for item in sublist]
    )
    save_path = os.path.join(
        save_dir, f"sample_idx{str(sub_idxs)}_ip{ip_adapt_scale}.jpg"
    )
    make_image_grid(grid_image, row_num, col_num).save(save_path)
    logger.info(f"Visualize in {save_path}")

    return img_save_paths


def entrypoint() -> None:
    fire.Fire(infer_pipe)


if __name__ == "__main__":
    entrypoint()
