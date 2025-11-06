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

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, EulerDiscreteScheduler
from huggingface_hub import snapshot_download
from kolors.models.controlnet import ControlNetModel
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models.unet_2d_condition import UNet2DConditionModel
from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import (
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from embodied_gen.models.text_model import download_kolors_weights
from embodied_gen.utils.log import logger

__all__ = [
    "build_texture_gen_pipe",
]


def build_texture_gen_pipe(
    base_ckpt_dir: str,
    controlnet_ckpt: str = None,
    ip_adapt_scale: float = 0,
    device: str = "cuda",
) -> DiffusionPipeline:
    """Build and initialize the Kolors + ControlNet (optional IP-Adapter) texture generation pipeline.

    Loads Kolors tokenizer, text encoder (ChatGLM), VAE, UNet, scheduler and (optionally)
    a ControlNet checkpoint plus IP-Adapter vision encoder. If ``controlnet_ckpt`` is
    not provided, the default multi-view texture ControlNet weights are downloaded
    automatically from the hub. When ``ip_adapt_scale > 0`` an IP-Adapter vision
    encoder and its weights are also loaded and activated.

    Args:
        base_ckpt_dir (str):
            Root directory where Kolors (and optionally Kolors-IP-Adapter-Plus) weights
            are or will be stored. Required subfolders: ``Kolors/{text_encoder,vae,unet,scheduler}``.
        controlnet_ckpt (str, optional):
            Directory containing a ControlNet checkpoint (safetensors). If ``None``,
            downloads the default ``texture_gen_mv_v1`` snapshot.
        ip_adapt_scale (float, optional):
            Strength (>=0) of IP-Adapter conditioning. Set >0 to enable IP-Adapter;
            typical values: 0.4-0.8. Default: 0 (disabled).
        device (str, optional):
            Target device to move the pipeline to (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
            Default: ``"cuda"``.

    Returns:
        DiffusionPipeline: A configured
        ``StableDiffusionXLControlNetImg2ImgPipeline`` ready for multi-view texture
        generation (with optional IP-Adapter support).

    Example:
        Initialize pipeline with IP-Adapter enabled.
        ```python
        from embodied_gen.models.texture_model import build_texture_gen_pipe
        ip_adapt_scale = 0.7
        PIPELINE = build_texture_gen_pipe(
            base_ckpt_dir="./weights",
            ip_adapt_scale=ip_adapt_scale,
            device="cuda",
        )
        PIPELINE.set_ip_adapter_scale([ip_adapt_scale])
        ```
        Initialize pipeline without IP-Adapter.
        ```python
        from embodied_gen.models.texture_model import build_texture_gen_pipe
        PIPELINE = build_texture_gen_pipe(
            base_ckpt_dir="./weights",
            ip_adapt_scale=0,
            device="cuda",
        )
        ```
    """

    download_kolors_weights(f"{base_ckpt_dir}/Kolors")
    logger.info(f"Load Kolors weights...")
    tokenizer = ChatGLMTokenizer.from_pretrained(
        f"{base_ckpt_dir}/Kolors/text_encoder"
    )
    text_encoder = ChatGLMModel.from_pretrained(
        f"{base_ckpt_dir}/Kolors/text_encoder", torch_dtype=torch.float16
    ).half()
    vae = AutoencoderKL.from_pretrained(
        f"{base_ckpt_dir}/Kolors/vae", revision=None
    ).half()
    unet = UNet2DConditionModel.from_pretrained(
        f"{base_ckpt_dir}/Kolors/unet", revision=None
    ).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(
        f"{base_ckpt_dir}/Kolors/scheduler"
    )

    if controlnet_ckpt is None:
        suffix = "texture_gen_mv_v1"  # "geo_cond_mv"
        model_path = snapshot_download(
            repo_id="xinjjj/RoboAssetGen", allow_patterns=f"{suffix}/*"
        )
        controlnet_ckpt = os.path.join(model_path, suffix)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_ckpt, use_safetensors=True
    ).half()

    # IP-Adapter model
    image_encoder = None
    clip_image_processor = None
    if ip_adapt_scale > 0:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            f"{base_ckpt_dir}/Kolors-IP-Adapter-Plus/image_encoder",
            # ignore_mismatched_sizes=True,
        ).to(dtype=torch.float16)
        ip_img_size = 336
        clip_image_processor = CLIPImageProcessor(
            size=ip_img_size, crop_size=ip_img_size
        )

    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=vae,
        controlnet=controlnet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False,
    )

    if ip_adapt_scale > 0:
        if hasattr(pipe.unet, "encoder_hid_proj"):
            pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj
        pipe.load_ip_adapter(
            f"{base_ckpt_dir}/Kolors-IP-Adapter-Plus",
            subfolder="",
            weight_name=["ip_adapter_plus_general.bin"],
        )
        pipe.set_ip_adapter_scale([ip_adapt_scale])

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()

    return pipe
