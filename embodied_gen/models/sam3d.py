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

from embodied_gen.utils.monkey_patches import monkey_patch_sam3d

monkey_patch_sam3d()
import os
import sys

import numpy as np
from hydra.utils import instantiate
from modelscope import snapshot_download
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, "../.."))
from loguru import logger
from thirdparty.sam3d.sam3d_objects.pipeline.inference_pipeline_pointmap import (
    InferencePipelinePointMap,
)

logger.remove()
logger.add(lambda _: None, level="ERROR")


__all__ = ["Sam3dInference"]


class Sam3dInference:
    """Wrapper for the SAM-3D-Objects inference pipeline.

    This class handles loading the SAM-3D-Objects model, configuring it for inference,
    and running the pipeline on input images (optionally with masks and pointmaps).
    It supports distillation options and inference step customization.

    Args:
        local_dir (str): Directory to store or load model weights and configs.
        compile (bool): Whether to compile the model for faster inference.

    Methods:
        merge_mask_to_rgba(image, mask):
            Merges a binary mask into the alpha channel of an RGB image.

        run(image, mask=None, seed=None, pointmap=None, use_stage1_distillation=False,
            use_stage2_distillation=False, stage1_inference_steps=25, stage2_inference_steps=25):
            Runs the inference pipeline and returns the output dictionary.
    """

    def __init__(
        self, local_dir: str = "weights/sam-3d-objects", compile: bool = False
    ) -> None:
        if not os.path.exists(local_dir):
            snapshot_download("facebook/sam-3d-objects", local_dir=local_dir)
        config_file = os.path.join(local_dir, "checkpoints/pipeline.yaml")
        config = OmegaConf.load(config_file)
        config.rendering_engine = "nvdiffrast"
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        # Generate 4 instead of 32 gs in each pixel for efficient storage.
        config["slat_decoder_gs_config_path"] = config.pop(
            "slat_decoder_gs_4_config_path", "slat_decoder_gs_4.yaml"
        )
        config["slat_decoder_gs_ckpt_path"] = config.pop(
            "slat_decoder_gs_4_ckpt_path", "slat_decoder_gs_4.ckpt"
        )
        self.pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)

        return rgba_image

    def run(
        self,
        image: np.ndarray | Image.Image,
        mask: np.ndarray = None,
        seed: int = None,
        pointmap: np.ndarray = None,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        stage1_inference_steps: int = 25,
        stage2_inference_steps: int = 25,
    ) -> dict:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if mask is not None:
            image = self.merge_mask_to_rgba(image, mask)
        return self.pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            use_stage1_distillation=use_stage1_distillation,
            use_stage2_distillation=use_stage2_distillation,
            stage1_inference_steps=stage1_inference_steps,
            stage2_inference_steps=stage2_inference_steps,
            pointmap=pointmap,
        )


if __name__ == "__main__":
    pipeline = Sam3dInference()

    from time import time

    import torch
    from embodied_gen.models.segment_model import RembgRemover

    input_image = "apps/assets/example_image/sample_00.jpg"
    output_gs = "outputs/splat.ply"
    remover = RembgRemover()
    clean_image = remover(input_image)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start = time()
    output = pipeline.run(clean_image, seed=42)
    print(f"Running cost: {round(time()-start, 1)}")

    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"(Max VRAM): {max_memory:.2f} GB")

    print(f"End: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    output["gs"].save_ply(output_gs)
    print(f"Saved to {output_gs}")
