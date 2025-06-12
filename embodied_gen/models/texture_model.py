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

__all__ = [
    "build_texture_gen_pipe",
]


def build_texture_gen_pipe(
    base_ckpt_dir: str,
    controlnet_ckpt: str = None,
    ip_adapt_scale: float = 0,
    device: str = "cuda",
) -> DiffusionPipeline:
    download_kolors_weights(f"{base_ckpt_dir}/Kolors")

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
        suffix = "geo_cond_mv"
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
