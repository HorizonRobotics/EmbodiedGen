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
# Text-to-Image generation models from Hugging Face community.

import os
from abc import ABC, abstractmethod

import torch
from diffusers import (
    ChromaPipeline,
    Cosmos2TextToImagePipeline,
    DPMSolverMultistepScheduler,
    FluxPipeline,
    KolorsPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.quantizers import PipelineQuantizationConfig
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, SiglipProcessor

__all__ = [
    "build_hf_image_pipeline",
]


class BasePipelineLoader(ABC):
    """Abstract base class for loading Hugging Face image generation pipelines.

    Attributes:
        device (str): Device to load the pipeline on.

    Methods:
        load(): Loads and returns the pipeline.
    """

    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def load(self):
        """Load and return the pipeline instance."""
        pass


class BasePipelineRunner(ABC):
    """Abstract base class for running image generation pipelines.

    Attributes:
        pipe: The loaded pipeline.

    Methods:
        run(prompt, **kwargs): Runs the pipeline with a prompt.
    """

    def __init__(self, pipe):
        self.pipe = pipe

    @abstractmethod
    def run(self, prompt: str, **kwargs) -> Image.Image:
        """Run the pipeline with the given prompt.

        Args:
            prompt (str): Text prompt for image generation.
            **kwargs: Additional pipeline arguments.

        Returns:
            Image.Image: Generated image(s).
        """
        pass


# ===== SD3.5-medium =====
class SD35Loader(BasePipelineLoader):
    """Loader for Stable Diffusion 3.5 medium pipeline."""

    def load(self):
        """Load the Stable Diffusion 3.5 medium pipeline.

        Returns:
            StableDiffusion3Pipeline: Loaded pipeline.
        """
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        return pipe


class SD35Runner(BasePipelineRunner):
    """Runner for Stable Diffusion 3.5 medium pipeline."""

    def run(self, prompt: str, **kwargs) -> Image.Image:
        """Generate images using Stable Diffusion 3.5 medium.

        Args:
            prompt (str): Text prompt.
            **kwargs: Additional arguments.

        Returns:
            Image.Image: Generated image(s).
        """
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Cosmos2 =====
class CosmosLoader(BasePipelineLoader):
    """Loader for Cosmos2 text-to-image pipeline."""

    def __init__(
        self,
        model_id="nvidia/Cosmos-Predict2-2B-Text2Image",
        local_dir="weights/cosmos2",
        device="cuda",
    ):
        super().__init__(device)
        self.model_id = model_id
        self.local_dir = local_dir

    def _patch(self):
        """Patch model and processor for optimized loading."""

        def patch_model(cls):
            orig = cls.from_pretrained

            def new(*args, **kwargs):
                kwargs.setdefault("attn_implementation", "flash_attention_2")
                kwargs.setdefault("torch_dtype", torch.bfloat16)
                return orig(*args, **kwargs)

            cls.from_pretrained = new

        def patch_processor(cls):
            orig = cls.from_pretrained

            def new(*args, **kwargs):
                kwargs.setdefault("use_fast", True)
                return orig(*args, **kwargs)

            cls.from_pretrained = new

        patch_model(AutoModelForCausalLM)
        patch_processor(SiglipProcessor)

    def load(self):
        """Load the Cosmos2 text-to-image pipeline.

        Returns:
            Cosmos2TextToImagePipeline: Loaded pipeline.
        """
        self._patch()
        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["text_encoder", "transformer", "unet"],
        )

        pipe = Cosmos2TextToImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=config,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        return pipe


class CosmosRunner(BasePipelineRunner):
    """Runner for Cosmos2 text-to-image pipeline."""

    def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
        """Generate images using Cosmos2 pipeline.

        Args:
            prompt (str): Text prompt.
            negative_prompt (str, optional): Negative prompt.
            **kwargs: Additional arguments.

        Returns:
            Image.Image: Generated image(s).
        """
        return self.pipe(
            prompt=prompt, negative_prompt=negative_prompt, **kwargs
        ).images


# ===== Kolors =====
class KolorsLoader(BasePipelineLoader):
    """Loader for Kolors pipeline."""

    def load(self):
        """Load the Kolors pipeline.

        Returns:
            KolorsPipeline: Loaded pipeline.
        """
        pipe = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        return pipe


class KolorsRunner(BasePipelineRunner):
    """Runner for Kolors pipeline."""

    def run(self, prompt: str, **kwargs) -> Image.Image:
        """Generate images using Kolors pipeline.

        Args:
            prompt (str): Text prompt.
            **kwargs: Additional arguments.

        Returns:
            Image.Image: Generated image(s).
        """
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Flux =====
class FluxLoader(BasePipelineLoader):
    """Loader for Flux pipeline."""

    def load(self):
        """Load the Flux pipeline.

        Returns:
            FluxPipeline: Loaded pipeline.
        """
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        return pipe.to(self.device)


class FluxRunner(BasePipelineRunner):
    """Runner for Flux pipeline."""

    def run(self, prompt: str, **kwargs) -> Image.Image:
        """Generate images using Flux pipeline.

        Args:
            prompt (str): Text prompt.
            **kwargs: Additional arguments.

        Returns:
            Image.Image: Generated image(s).
        """
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Chroma =====
class ChromaLoader(BasePipelineLoader):
    """Loader for Chroma pipeline."""

    def load(self):
        """Load the Chroma pipeline.

        Returns:
            ChromaPipeline: Loaded pipeline.
        """
        return ChromaPipeline.from_pretrained(
            "lodestones/Chroma", torch_dtype=torch.bfloat16
        ).to(self.device)


class ChromaRunner(BasePipelineRunner):
    """Runner for Chroma pipeline."""

    def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
        """Generate images using Chroma pipeline.

        Args:
            prompt (str): Text prompt.
            negative_prompt (str, optional): Negative prompt.
            **kwargs: Additional arguments.

        Returns:
            Image.Image: Generated image(s).
        """
        return self.pipe(
            prompt=prompt, negative_prompt=negative_prompt, **kwargs
        ).images


PIPELINE_REGISTRY = {
    "sd35": (SD35Loader, SD35Runner),
    "cosmos": (CosmosLoader, CosmosRunner),
    "kolors": (KolorsLoader, KolorsRunner),
    "flux": (FluxLoader, FluxRunner),
    "chroma": (ChromaLoader, ChromaRunner),
}


def build_hf_image_pipeline(name: str, device="cuda") -> BasePipelineRunner:
    """Build a Hugging Face image generation pipeline runner by name.

    Args:
        name (str): Name of the pipeline (e.g., "sd35", "cosmos").
        device (str): Device to load the pipeline on.

    Returns:
        BasePipelineRunner: Pipeline runner instance.

    Example:
        ```py
        from embodied_gen.models.image_comm_model import build_hf_image_pipeline
        runner = build_hf_image_pipeline("sd35")
        images = runner.run(prompt="A robot holding a sign that says 'Hello'")
        ```
    """
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unsupported model: {name}")
    loader_cls, runner_cls = PIPELINE_REGISTRY[name]
    pipe = loader_cls(device=device).load()

    return runner_cls(pipe)


if __name__ == "__main__":
    model_name = "sd35"
    runner = build_hf_image_pipeline(model_name)
    # NOTE: Just for pipeline testing, generation quality at low resolution is poor.
    images = runner.run(
        prompt="A robot holding a sign that says 'Hello'",
        height=512,
        width=512,
        num_inference_steps=10,
        guidance_scale=6,
        num_images_per_prompt=1,
    )

    for i, img in enumerate(images):
        img.save(f"image_{model_name}_{i}.jpg")
