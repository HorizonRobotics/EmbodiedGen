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
from typing import Union

import cv2
import numpy as np
import spaces
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from huggingface_hub import snapshot_download
from PIL import Image
from embodied_gen.models.segment_model import RembgRemover

__all__ = [
    "DelightingModel",
]


class DelightingModel(object):
    """A model to remove the lighting in image space.

    This model is encapsulated based on the Hunyuan3D-Delight model
    from https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0 # noqa

    Attributes:
        image_guide_scale (float): Weight of image guidance in diffusion process.
        text_guide_scale (float): Weight of text (prompt) guidance in diffusion process.
        num_infer_step (int): Number of inference steps for diffusion model.
        mask_erosion_size (int): Size of erosion kernel for alpha mask cleanup.
        device (str): Device used for inference, e.g., 'cuda' or 'cpu'.
        seed (int): Random seed for diffusion model reproducibility.
        model_path (str): Filesystem path to pretrained model weights.
        pipeline: Lazy-loaded diffusion pipeline instance.
    """

    def __init__(
        self,
        model_path: str = None,
        num_infer_step: int = 50,
        mask_erosion_size: int = 3,
        image_guide_scale: float = 1.5,
        text_guide_scale: float = 1.0,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.image_guide_scale = image_guide_scale
        self.text_guide_scale = text_guide_scale
        self.num_infer_step = num_infer_step
        self.mask_erosion_size = mask_erosion_size
        self.kernel = np.ones(
            (self.mask_erosion_size, self.mask_erosion_size), np.uint8
        )
        self.seed = seed
        self.device = device
        self.pipeline = None  # lazy load model adapt to @spaces.GPU

        if model_path is None:
            suffix = "hunyuan3d-delight-v2-0"
            model_path = snapshot_download(
                repo_id="tencent/Hunyuan3D-2", allow_patterns=f"{suffix}/*"
            )
            model_path = os.path.join(model_path, suffix)

        self.model_path = model_path

    def _lazy_init_pipeline(self):
        if self.pipeline is None:
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline.set_progress_bar_config(disable=True)

            pipeline.to(self.device, torch.float16)
            self.pipeline = pipeline

    def recenter_image(
        self, image: Image.Image, border_ratio: float = 0.2
    ) -> Image.Image:
        if image.mode == "RGB":
            return image
        elif image.mode == "L":
            image = image.convert("RGB")
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop(
            (min_col, min_row, max_col + 1, max_row + 1)
        )

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new(
            "RGBA", (square_size, square_size), (255, 255, 255, 0)
        )

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))

        return new_image

    @spaces.GPU
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        preprocess: bool = False,
        target_wh: tuple[int, int] = None,
    ) -> Image.Image:
        self._lazy_init_pipeline()

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if preprocess:
            bg_remover = RembgRemover()
            image = bg_remover(image)
            image = self.recenter_image(image)

        if target_wh is not None:
            image = image.resize(target_wh)
        else:
            target_wh = image.size

        image_array = np.array(image)
        assert image_array.shape[-1] == 4, "Image must have alpha channel"

        raw_alpha_channel = image_array[:, :, 3]
        alpha_channel = cv2.erode(raw_alpha_channel, self.kernel, iterations=1)
        image_array[alpha_channel == 0, :3] = 255  # must be white background
        image_array[:, :, 3] = alpha_channel

        image = self.pipeline(
            prompt="",
            image=Image.fromarray(image_array).convert("RGB"),
            generator=torch.manual_seed(self.seed),
            num_inference_steps=self.num_infer_step,
            image_guidance_scale=self.image_guide_scale,
            guidance_scale=self.text_guide_scale,
        ).images[0]

        alpha_channel = Image.fromarray(alpha_channel)
        rgba_image = image.convert("RGBA").resize(target_wh)
        rgba_image.putalpha(alpha_channel)

        return rgba_image


if __name__ == "__main__":
    delighting_model = DelightingModel()
    image_path = "apps/assets/example_image/sample_12.jpg"
    image = delighting_model(
        image_path, preprocess=True, target_wh=(512, 512)
    )  # noqa
    image.save("delight.png")

    # image_path = "embodied_gen/scripts/test_robot.png"
    # image = delighting_model(image_path)
    # image.save("delighting_image_a2.png")
