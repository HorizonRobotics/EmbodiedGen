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
from typing import Literal, Union

import cv2
import numpy as np
import rembg
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)
from transformers import pipeline
from embodied_gen.data.utils import resize_pil, trellis_preprocess
from embodied_gen.utils.process_media import filter_small_connected_components
from embodied_gen.validators.quality_checkers import ImageSegChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "SAMRemover",
    "SAMPredictor",
    "RembgRemover",
    "get_segmented_image_by_agent",
]


class SAMRemover(object):
    """Loading SAM models and performing background removal on images.

    Attributes:
        checkpoint (str): Path to the model checkpoint.
        model_type (str): Type of the SAM model to load (default: "vit_h").
        area_ratio (float): Area ratio filtering small connected components.
    """

    def __init__(
        self,
        checkpoint: str = None,
        model_type: str = "vit_h",
        area_ratio: float = 15,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.area_ratio = area_ratio

        if checkpoint is None:
            suffix = "sam"
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns=f"{suffix}/*"
            )
            checkpoint = os.path.join(
                model_path, suffix, "sam_vit_h_4b8939.pth"
            )

        self.mask_generator = self._load_sam_model(checkpoint)

    def _load_sam_model(self, checkpoint: str) -> SamAutomaticMaskGenerator:
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        sam.to(device=self.device)

        return SamAutomaticMaskGenerator(sam)

    def __call__(
        self, image: Union[str, Image.Image, np.ndarray], save_path: str = None
    ) -> Image.Image:
        """Removes the background from an image using the SAM model.

        Args:
            image (Union[str, Image.Image, np.ndarray]): Input image,
                can be a file path, PIL Image, or numpy array.
            save_path (str): Path to save the output image (default: None).

        Returns:
            Image.Image: The image with background removed,
                including an alpha channel.
        """
        # Convert input to numpy array
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        image = resize_pil(image)
        image = np.array(image.convert("RGB"))

        # Generate masks
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        if not masks:
            logger.warning(
                "Segmentation failed: No mask generated, return raw image."
            )
            output_image = Image.fromarray(image, mode="RGB")
        else:
            # Use the largest mask
            best_mask = masks[0]["segmentation"]
            mask = (best_mask * 255).astype(np.uint8)
            mask = filter_small_connected_components(
                mask, area_ratio=self.area_ratio
            )
            # Apply the mask to remove the background
            background_removed = cv2.bitwise_and(image, image, mask=mask)
            output_image = np.dstack((background_removed, mask))
            output_image = Image.fromarray(output_image, mode="RGBA")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            output_image.save(save_path)

        return output_image


class SAMPredictor(object):
    def __init__(
        self,
        checkpoint: str = None,
        model_type: str = "vit_h",
        binary_thresh: float = 0.1,
        device: str = "cuda",
    ):
        self.device = device
        self.model_type = model_type

        if checkpoint is None:
            suffix = "sam"
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns=f"{suffix}/*"
            )
            checkpoint = os.path.join(
                model_path, suffix, "sam_vit_h_4b8939.pth"
            )

        self.predictor = self._load_sam_model(checkpoint)
        self.binary_thresh = binary_thresh

    def _load_sam_model(self, checkpoint: str) -> SamPredictor:
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        sam.to(device=self.device)

        return SamPredictor(sam)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        image = resize_pil(image)
        image = np.array(image.convert("RGB"))

        return image

    def generate_masks(
        self,
        image: np.ndarray,
        selected_points: list[list[int]],
    ) -> np.ndarray:
        if len(selected_points) == 0:
            return []

        points = (
            torch.Tensor([p for p, _ in selected_points])
            .to(self.predictor.device)
            .unsqueeze(1)
        )

        labels = (
            torch.Tensor([int(l) for _, l in selected_points])
            .to(self.predictor.device)
            .unsqueeze(1)
        )

        transformed_points = self.predictor.transform.apply_coords_torch(
            points, image.shape[:2]
        )

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=transformed_points,
            point_labels=labels,
            multimask_output=True,
        )
        valid_mask = masks[:, torch.argmax(scores, dim=1)]
        masks_pos = valid_mask[labels[:, 0] == 1, 0].cpu().detach().numpy()
        masks_neg = valid_mask[labels[:, 0] == 0, 0].cpu().detach().numpy()
        if len(masks_neg) == 0:
            masks_neg = np.zeros_like(masks_pos)
        if len(masks_pos) == 0:
            masks_pos = np.zeros_like(masks_neg)
        masks_neg = masks_neg.max(axis=0, keepdims=True)
        masks_pos = masks_pos.max(axis=0, keepdims=True)
        valid_mask = (masks_pos.astype(int) - masks_neg.astype(int)).clip(0, 1)

        binary_mask = (valid_mask > self.binary_thresh).astype(np.int32)

        return [(mask, f"mask_{i}") for i, mask in enumerate(binary_mask)]

    def get_segmented_image(
        self, image: np.ndarray, masks: list[tuple[np.ndarray, str]]
    ) -> Image.Image:
        seg_image = Image.fromarray(image, mode="RGB")
        alpha_channel = np.zeros(
            (seg_image.height, seg_image.width), dtype=np.uint8
        )
        for mask, _ in masks:
            # Use the maximum to combine multiple masks
            alpha_channel = np.maximum(alpha_channel, mask)

        alpha_channel = np.clip(alpha_channel, 0, 1)
        alpha_channel = (alpha_channel * 255).astype(np.uint8)
        alpha_image = Image.fromarray(alpha_channel, mode="L")
        r, g, b = seg_image.split()
        seg_image = Image.merge("RGBA", (r, g, b, alpha_image))

        return seg_image

    def __call__(
        self,
        image: Union[str, Image.Image, np.ndarray],
        selected_points: list[list[int]],
    ) -> Image.Image:
        image = self.preprocess_image(image)
        self.predictor.set_image(image)
        masks = self.generate_masks(image, selected_points)

        return self.get_segmented_image(image, masks)


class RembgRemover(object):
    def __init__(self):
        self.rembg_session = rembg.new_session("u2net")

    def __call__(
        self, image: Union[str, Image.Image, np.ndarray], save_path: str = None
    ) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = resize_pil(image)
        output_image = rembg.remove(image, session=self.rembg_session)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            output_image.save(save_path)

        return output_image


class BMGG14Remover(object):
    def __init__(self) -> None:
        self.model = pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
        )

    def __call__(
        self, image: Union[str, Image.Image, np.ndarray], save_path: str = None
    ):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = resize_pil(image)
        output_image = self.model(image)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            output_image.save(save_path)

        return output_image


def invert_rgba_pil(
    image: Image.Image, mask: Image.Image, save_path: str = None
) -> Image.Image:
    mask = (255 - np.array(mask))[..., None]
    image_array = np.concatenate([np.array(image), mask], axis=-1)
    inverted_image = Image.fromarray(image_array, "RGBA")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        inverted_image.save(save_path)

    return inverted_image


def get_segmented_image_by_agent(
    image: Image.Image,
    sam_remover: SAMRemover,
    rbg_remover: RembgRemover,
    seg_checker: ImageSegChecker = None,
    save_path: str = None,
    mode: Literal["loose", "strict"] = "loose",
) -> Image.Image:
    def _is_valid_seg(raw_img: Image.Image, seg_img: Image.Image) -> bool:
        if seg_checker is None:
            return True
        return raw_img.mode == "RGBA" and seg_checker([raw_img, seg_img])[0]

    out_sam = f"{save_path}_sam.png" if save_path else None
    out_sam_inv = f"{save_path}_sam_inv.png" if save_path else None
    out_rbg = f"{save_path}_rbg.png" if save_path else None

    seg_image = sam_remover(image, out_sam)
    seg_image = seg_image.convert("RGBA")
    _, _, _, alpha = seg_image.split()
    seg_image_inv = invert_rgba_pil(image.convert("RGB"), alpha, out_sam_inv)
    seg_image_rbg = rbg_remover(image, out_rbg)

    final_image = None
    if _is_valid_seg(image, seg_image):
        final_image = seg_image
    elif _is_valid_seg(image, seg_image_inv):
        final_image = seg_image_inv
    elif _is_valid_seg(image, seg_image_rbg):
        logger.warning(f"Failed to segment by `SAM`, retry with `rembg`.")
        final_image = seg_image_rbg
    else:
        if mode == "strict":
            raise RuntimeError(
                f"Failed to segment by `SAM` or `rembg`, abort."
            )
        logger.warning("Failed to segment by SAM or rembg, use raw image.")
        final_image = image.convert("RGBA")

    if save_path:
        final_image.save(save_path)

    final_image = trellis_preprocess(final_image)

    return final_image


if __name__ == "__main__":
    input_image = "outputs/text2image/demo_objects/electrical/sample_0.jpg"
    output_image = "sample_0_seg2.png"

    # input_image = "outputs/text2image/tmp/coffee_machine.jpeg"
    # output_image = "outputs/text2image/tmp/coffee_machine_seg.png"

    # input_image = "outputs/text2image/tmp/bucket.jpeg"
    # output_image = "outputs/text2image/tmp/bucket_seg.png"

    remover = SAMRemover(model_type="vit_h")
    remover = RembgRemover()
    clean_image = remover(input_image)
    clean_image.save(output_image)
    get_segmented_image_by_agent(
        Image.open(input_image), remover, remover, None, "./test_seg.png"
    )

    remover = BMGG14Remover()
    remover("embodied_gen/models/test_seg.jpg", "./seg.png")
