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
import random

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models.unet_2d_condition import (
    UNet2DConditionModel as UNet2DConditionModelIP,
)
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import (
    StableDiffusionXLPipeline,
)
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import (  # noqa
    StableDiffusionXLPipeline as StableDiffusionXLPipelineIP,
)
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "build_text2img_ip_pipeline",
    "build_text2img_pipeline",
    "text2img_gen",
]


def build_text2img_ip_pipeline(
    ckpt_dir: str,
    ref_scale: float,
    device: str = "cuda",
) -> StableDiffusionXLPipelineIP:
    text_encoder = ChatGLMModel.from_pretrained(
        f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
    ).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")
    vae = AutoencoderKL.from_pretrained(
        f"{ckpt_dir}/vae", revision=None
    ).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModelIP.from_pretrained(
        f"{ckpt_dir}/unet", revision=None
    ).half()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        f"{ckpt_dir}/../Kolors-IP-Adapter-Plus/image_encoder",
        ignore_mismatched_sizes=True,
    ).to(dtype=torch.float16)
    clip_image_processor = CLIPImageProcessor(size=336, crop_size=336)

    pipe = StableDiffusionXLPipelineIP(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False,
    )

    if hasattr(pipe.unet, "encoder_hid_proj"):
        pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj

    pipe.load_ip_adapter(
        f"{ckpt_dir}/../Kolors-IP-Adapter-Plus",
        subfolder="",
        weight_name=["ip_adapter_plus_general.bin"],
    )
    pipe.set_ip_adapter_scale([ref_scale])

    pipe = pipe.to(device)
    pipe.image_encoder = pipe.image_encoder.to(device)
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_vae_slicing()

    return pipe


def build_text2img_pipeline(
    ckpt_dir: str,
    device: str = "cuda",
) -> StableDiffusionXLPipeline:
    text_encoder = ChatGLMModel.from_pretrained(
        f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
    ).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")
    vae = AutoencoderKL.from_pretrained(
        f"{ckpt_dir}/vae", revision=None
    ).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        f"{ckpt_dir}/unet", revision=None
    ).half()
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False,
    )
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def text2img_gen(
    prompt: str,
    n_sample: int,
    guidance_scale: float,
    pipeline: StableDiffusionXLPipeline | StableDiffusionXLPipelineIP,
    ip_image: Image.Image | str = None,
    image_wh: tuple[int, int] = [1024, 1024],
    infer_step: int = 50,
    ip_image_size: int = 512,
    seed: int = None,
) -> list[Image.Image]:
    prompt = "Single " + prompt + ", in the center of the image"
    prompt += ", high quality, high resolution, best quality, white background, 3D style"  # noqa
    logger.info(f"Processing prompt: {prompt}")

    generator = None
    if seed is not None:
        generator = torch.Generator(pipeline.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    kwargs = dict(
        prompt=prompt,
        height=image_wh[1],
        width=image_wh[0],
        num_inference_steps=infer_step,
        guidance_scale=guidance_scale,
        num_images_per_prompt=n_sample,
        generator=generator,
    )
    if ip_image is not None:
        if isinstance(ip_image, str):
            ip_image = Image.open(ip_image)
        ip_image = ip_image.resize((ip_image_size, ip_image_size))
        kwargs.update(ip_adapter_image=[ip_image])

    return pipeline(**kwargs).images
