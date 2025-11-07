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
from typing import Union

import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from embodied_gen.data.utils import get_images_from_grid

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


__all__ = [
    "ImageStableSR",
    "ImageRealESRGAN",
]


class ImageStableSR:
    """Super-resolution image upscaler using Stable Diffusion x4 upscaling model.

    This class wraps the StabilityAI Stable Diffusion x4 upscaler for high-quality
    image super-resolution.

    Args:
        model_path (str, optional): Path or HuggingFace repo for the model.
        device (str, optional): Device for inference.

    Example:
        ```py
        from embodied_gen.models.sr_model import ImageStableSR
        from PIL import Image

        sr_model = ImageStableSR()
        img = Image.open("input.png")
        upscaled = sr_model(img)
        upscaled.save("output.png")
        ```
    """

    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-x4-upscaler",
        device="cuda",
    ) -> None:
        """Initializes the Stable Diffusion x4 upscaler.

        Args:
            model_path (str, optional): Model path or repo.
            device (str, optional): Device for inference.
        """
        from diffusers import StableDiffusionUpscalePipeline

        self.up_pipeline_x4 = StableDiffusionUpscalePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        self.up_pipeline_x4.set_progress_bar_config(disable=True)
        self.up_pipeline_x4.enable_model_cpu_offload()

    @spaces.GPU
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str = "",
        infer_step: int = 20,
    ) -> Image.Image:
        """Performs super-resolution on the input image.

        Args:
            image (Union[Image.Image, np.ndarray]): Input image.
            prompt (str, optional): Text prompt for upscaling.
            infer_step (int, optional): Number of inference steps.

        Returns:
            Image.Image: Upscaled image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        with torch.no_grad():
            upscaled_image = self.up_pipeline_x4(
                image=image,
                prompt=[prompt],
                num_inference_steps=infer_step,
            ).images[0]

        return upscaled_image


class ImageRealESRGAN:
    """A wrapper for Real-ESRGAN-based image super-resolution.

    This class uses the RealESRGAN model to perform image upscaling,
    typically by a factor of 4.

    Attributes:
        outscale (int): The output image scale factor (e.g., 2, 4).
        model_path (str): Path to the pre-trained model weights.

    Example:
        ```py
        from embodied_gen.models.sr_model import ImageRealESRGAN
        from PIL import Image

        sr_model = ImageRealESRGAN(outscale=4)
        img = Image.open("input.png")
        upscaled = sr_model(img)
        upscaled.save("output.png")
        ```
    """

    def __init__(self, outscale: int, model_path: str = None) -> None:
        """Initializes the RealESRGAN upscaler.

        Args:
            outscale (int): Output scale factor.
            model_path (str, optional): Path to model weights.
        """
        # monkey patch to support torchvision>=0.16
        import torchvision
        from packaging import version

        if version.parse(torchvision.__version__) > version.parse("0.16"):
            import sys
            import types

            import torchvision.transforms.functional as TF

            functional_tensor = types.ModuleType(
                "torchvision.transforms.functional_tensor"
            )
            functional_tensor.rgb_to_grayscale = TF.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = (
                functional_tensor
            )

        self.outscale = outscale
        self.upsampler = None

        if model_path is None:
            suffix = "super_resolution"
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns=f"{suffix}/*"
            )
            model_path = os.path.join(
                model_path, suffix, "RealESRGAN_x4plus.pth"
            )

        self.model_path = model_path

    def _lazy_init(self):
        """Lazily initializes the RealESRGAN model."""
        if self.upsampler is None:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

            self.upsampler = RealESRGANer(
                scale=4,
                model_path=self.model_path,
                model=model,
                pre_pad=0,
                half=True,
            )

    @spaces.GPU
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Performs super-resolution on the input image.

        Args:
            image (Union[Image.Image, np.ndarray]): Input image.

        Returns:
            Image.Image: Upscaled image.
        """
        self._lazy_init()

        if isinstance(image, Image.Image):
            image = np.array(image)

        with torch.no_grad():
            output, _ = self.upsampler.enhance(image, outscale=self.outscale)

        return Image.fromarray(output)


if __name__ == "__main__":
    color_path = "outputs/texture_mesh_gen/multi_view/color_sample0.png"

    # Use RealESRGAN_x4plus for x4 (512->2048) image super resolution.
    super_model = ImageRealESRGAN(outscale=4)
    multiviews = get_images_from_grid(color_path, img_size=512)
    multiviews = [super_model(img.convert("RGB")) for img in multiviews]
    for idx, img in enumerate(multiviews):
        img.save(f"sr{idx}.png")

    # # Use stable diffusion for x4 (512->2048) image super resolution.
    # super_model = ImageStableSR()
    # multiviews = get_images_from_grid(color_path, img_size=512)
    # multiviews = [super_model(img) for img in multiviews]
    # for idx, img in enumerate(multiviews):
    #     img.save(f"sr_stable{idx}.png")
