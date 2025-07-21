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


import json
import logging
import os
import random
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


__all__ = [
    "Asset3dGenDataset",
    "PanoGSplatDataset",
]


class Asset3dGenDataset(Dataset):
    def __init__(
        self,
        index_file: str,
        target_hw: Tuple[int, int],
        transform: Callable = None,
        control_transform: Callable = None,
        max_train_samples: int = None,
        sub_idxs: List[List[int]] = None,
        seed: int = 79,
    ) -> None:
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"{index_file} index_file not found.")

        self.index_file = index_file
        self.target_hw = target_hw
        self.transform = transform
        self.control_transform = control_transform
        self.max_train_samples = max_train_samples
        self.meta_info = self.prepare_data_index(index_file)
        self.data_list = sorted(self.meta_info.keys())
        self.sub_idxs = sub_idxs  # sub_idxs [[0,1,2], [3,4,5], [...], ...]
        self.image_num = 6  # hardcode temp.
        random.seed(seed)
        logger.info(f"Trainset: {len(self)} asset3d instances.")

    def __len__(self) -> int:
        return len(self.meta_info)

    def prepare_data_index(self, index_file: str) -> Dict[str, Any]:
        with open(index_file, "r") as fin:
            meta_info = json.load(fin)

        meta_info_filtered = dict()
        for idx, uid in enumerate(meta_info):
            if "status" not in meta_info[uid]:
                continue
            if meta_info[uid]["status"] != "success":
                continue
            if self.max_train_samples and idx >= self.max_train_samples:
                break

            meta_info_filtered[uid] = meta_info[uid]

        logger.info(
            f"Load {len(meta_info)} assets, keep {len(meta_info_filtered)} valids."  # noqa
        )

        return meta_info_filtered

    def fetch_sample_images(
        self,
        uid: str,
        attrs: List[str],
        sub_index: int = None,
        transform: Callable = None,
    ) -> torch.Tensor:
        sample = self.meta_info[uid]
        images = []
        for attr in attrs:
            item = sample[attr]
            if sub_index is not None:
                item = item[sub_index]
            mode = "L" if attr == "image_mask" else "RGB"
            image = Image.open(item).convert(mode)
            if transform is not None:
                image = transform(image)
                if len(image.shape) == 2:
                    image = image[..., None]
            images.append(image)

        images = torch.cat(images, dim=0)

        return images

    def fetch_sample_grid_images(
        self,
        uid: str,
        attrs: List[str],
        sub_idxs: List[List[int]],
        transform: Callable = None,
    ) -> torch.Tensor:
        assert transform is not None

        grid_image = []
        for row_idxs in sub_idxs:
            row_image = []
            for row_idx in row_idxs:
                image = self.fetch_sample_images(
                    uid, attrs, row_idx, transform
                )
                row_image.append(image)
            row_image = torch.cat(row_image, dim=2)  # (c h w)
            grid_image.append(row_image)

        grid_image = torch.cat(grid_image, dim=1)

        return grid_image

    def compute_text_embeddings(
        self, embed_path: str, original_size: Tuple[int, int]
    ) -> Dict[str, nn.Module]:
        data_dict = torch.load(embed_path)
        prompt_embeds = data_dict["prompt_embeds"][0]
        add_text_embeds = data_dict["pooled_prompt_embeds"][0]

        # Need changed if random crop, set as crop_top_left [y1, x1], center crop as [0, 0].  # noqa
        crops_coords_top_left = (0, 0)
        add_time_ids = list(
            original_size + crops_coords_top_left + self.target_hw
        )
        add_time_ids = torch.tensor([add_time_ids])
        # add_time_ids = add_time_ids.repeat((len(add_text_embeds), 1))

        unet_added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    def visualize_item(
        self,
        control: torch.Tensor,
        color: torch.Tensor,
        save_dir: str = None,
    ) -> List[Image.Image]:
        to_pil = transforms.ToPILImage()

        color = (color + 1) / 2
        color_pil = to_pil(color)
        normal_pil = to_pil(control[0:3])
        position_pil = to_pil(control[3:6])
        mask_pil = to_pil(control[6:])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            color_pil.save(f"{save_dir}/rgb.jpg")
            normal_pil.save(f"{save_dir}/normal.jpg")
            position_pil.save(f"{save_dir}/position.jpg")
            mask_pil.save(f"{save_dir}/mask.jpg")
            logger.info(f"Visualization in {save_dir}")

        return normal_pil, position_pil, mask_pil, color_pil

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uid = self.data_list[index]

        sub_idxs = self.sub_idxs
        if sub_idxs is None:
            sub_idxs = [[random.randint(0, self.image_num - 1)]]

        input_image = self.fetch_sample_grid_images(
            uid,
            attrs=["image_view_normal", "image_position", "image_mask"],
            sub_idxs=sub_idxs,
            transform=self.control_transform,
        )
        assert input_image.shape[1:] == self.target_hw

        output_image = self.fetch_sample_grid_images(
            uid,
            attrs=["image_color"],
            sub_idxs=sub_idxs,
            transform=self.transform,
        )

        sample = self.meta_info[uid]
        text_feats = self.compute_text_embeddings(
            sample["text_feat"], tuple(sample["image_hw"])
        )

        data = dict(
            pixel_values=output_image,
            conditioning_pixel_values=input_image,
            prompt_embeds=text_feats["prompt_embeds"],
            text_embeds=text_feats["text_embeds"],
            time_ids=text_feats["time_ids"],
        )

        return data


class PanoGSplatDataset(Dataset):
    """A PyTorch Dataset for loading panorama-based 3D Gaussian Splatting data.

    This dataset is designed to be compatible with train and eval pipelines
    that use COLMAP-style camera conventions.

    Args:
        data_dir (str): Root directory where the dataset file is located.
        split (str): Dataset split to use, either "train" or "eval".
        data_name (str, optional): Name of the dataset file (default: "gs_data.pt").
        max_sample_num (int, optional): Maximum number of samples to load. If None,
            all available samples in the split will be used.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = Literal["train", "eval"],
        data_name: str = "gs_data.pt",
        max_sample_num: int = None,
    ) -> None:
        self.data_path = os.path.join(data_dir, data_name)
        self.split = split
        self.max_sample_num = max_sample_num
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset file {self.data_path} not found. Please provide the correct path."
            )
        self.data = torch.load(self.data_path, weights_only=False)
        self.frames = self.data[split]
        if max_sample_num is not None:
            self.frames = self.frames[:max_sample_num]
        self.points = self.data.get("points", None)
        self.points_rgb = self.data.get("points_rgb", None)

    def __len__(self) -> int:
        return len(self.frames)

    def cvt_blender_to_colmap_coord(self, c2w: np.ndarray) -> np.ndarray:
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        tranformed_c2w = np.copy(c2w)
        tranformed_c2w[:3, 1:3] *= -1

        return tranformed_c2w

    def __getitem__(self, index: int) -> dict[str, any]:
        data = self.frames[index]
        c2w = self.cvt_blender_to_colmap_coord(data["camtoworld"])
        item = dict(
            camtoworld=c2w,
            K=data["K"],
            image_h=data["image_h"],
            image_w=data["image_w"],
        )
        if "image" in data:
            item["image"] = data["image"]
        if "image_id" in data:
            item["image_id"] = data["image_id"]

        return item


if __name__ == "__main__":
    index_file = "datasets/objaverse/v1.0/statistics_1.0_gobjaverse_filter/view6s_v4/meta_ac2e0ddea8909db26d102c8465b5bcb2.json"  # noqa
    target_hw = (512, 512)
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

    sub_idxs = [[0, 1, 2], [3, 4, 5]]  # None
    if sub_idxs is not None:
        target_hw = (
            target_hw[0] * len(sub_idxs),
            target_hw[1] * len(sub_idxs[0]),
        )

    dataset = Asset3dGenDataset(
        index_file,
        target_hw,
        image_transform,
        control_transform,
        sub_idxs=sub_idxs,
    )
    data = dataset[0]
    dataset.visualize_item(
        data["conditioning_pixel_values"], data["pixel_values"], save_dir="./"
    )
