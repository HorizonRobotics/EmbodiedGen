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

import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from PIL import Image


class AestheticPredictor:
    """Aesthetic Score Predictor using CLIP and a pre-trained MLP.

    Checkpoints from `https://github.com/christophschuhmann/improved-aesthetic-predictor/tree/main`.

    Args:
        clip_model_dir (str, optional): Path to CLIP model directory.
        sac_model_path (str, optional): Path to SAC model weights.
        device (str, optional): Device for computation ("cuda" or "cpu").

    Example:
        ```py
        from embodied_gen.validators.aesthetic_predictor import AestheticPredictor
        predictor = AestheticPredictor(device="cuda")
        score = predictor.predict("image.png")
        print("Aesthetic score:", score)
        ```
    """

    def __init__(self, clip_model_dir=None, sac_model_path=None, device="cpu"):

        self.device = device

        if clip_model_dir is None:
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns="aesthetic/*"
            )
            suffix = "aesthetic"
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns=f"{suffix}/*"
            )
            clip_model_dir = os.path.join(model_path, suffix)

        if sac_model_path is None:
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns="aesthetic/*"
            )
            suffix = "aesthetic"
            model_path = snapshot_download(
                repo_id="xinjjj/RoboAssetGen", allow_patterns=f"{suffix}/*"
            )
            sac_model_path = os.path.join(
                model_path, suffix, "sac+logos+ava1-l14-linearMSE.pth"
            )

        self.clip_model, self.preprocess = self._load_clip_model(
            clip_model_dir
        )
        self.sac_model = self._load_sac_model(sac_model_path, input_size=768)

    class MLP(pl.LightningModule):  # noqa
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.layers(x)

    @staticmethod
    def normalized(a, axis=-1, order=2):
        """Normalize the array to unit norm."""
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def _load_clip_model(self, model_dir: str, model_name: str = "ViT-L/14"):
        """Load the CLIP model."""
        model, preprocess = clip.load(
            model_name, download_root=model_dir, device=self.device
        )
        return model, preprocess

    def _load_sac_model(self, model_path, input_size):
        """Load the SAC model."""
        model = self.MLP(input_size)
        ckpt = torch.load(model_path, weights_only=True)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        """Predicts the aesthetic score for a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            float: Predicted aesthetic score.
        """
        if isinstance(image_path, str):
            pil_image = Image.open(image_path)
        else:
            pil_image = image_path

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract CLIP features
            image_features = self.clip_model.encode_image(image)
            # Normalize features
            normalized_features = self.normalized(
                image_features.cpu().detach().numpy()
            )
            # Predict score
            prediction = self.sac_model(
                torch.from_numpy(normalized_features)
                .type(torch.FloatTensor)
                .to(self.device)
            )

        return prediction.item()
