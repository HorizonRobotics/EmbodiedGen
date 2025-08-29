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
import sys
import zipfile

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms


def monkey_patch_pano2room():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    sys.path.append(os.path.join(current_dir, "../.."))
    sys.path.append(os.path.join(current_dir, "../../thirdparty/pano2room"))
    from thirdparty.pano2room.modules.geo_predictors.omnidata.omnidata_normal_predictor import (
        OmnidataNormalPredictor,
    )
    from thirdparty.pano2room.modules.geo_predictors.omnidata.omnidata_predictor import (
        OmnidataPredictor,
    )

    def patched_omni_depth_init(self):
        self.img_size = 384
        self.model = torch.hub.load(
            'alexsax/omnidata_models', 'depth_dpt_hybrid_384'
        )
        self.model.eval()
        self.trans_totensor = transforms.Compose(
            [
                transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
                transforms.CenterCrop(self.img_size),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    OmnidataPredictor.__init__ = patched_omni_depth_init

    def patched_omni_normal_init(self):
        self.img_size = 384
        self.model = torch.hub.load(
            'alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384'
        )
        self.model.eval()
        self.trans_totensor = transforms.Compose(
            [
                transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
                transforms.CenterCrop(self.img_size),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    OmnidataNormalPredictor.__init__ = patched_omni_normal_init

    def patched_panojoint_init(self, save_path=None):
        self.depth_predictor = OmnidataPredictor()
        self.normal_predictor = OmnidataNormalPredictor()
        self.save_path = save_path

    from modules.geo_predictors import PanoJointPredictor

    PanoJointPredictor.__init__ = patched_panojoint_init

    # NOTE: We use gsplat instead.
    # import depth_diff_gaussian_rasterization_min as ddgr
    # from dataclasses import dataclass
    # @dataclass
    # class PatchedGaussianRasterizationSettings:
    #     image_height: int
    #     image_width: int
    #     tanfovx: float
    #     tanfovy: float
    #     bg: torch.Tensor
    #     scale_modifier: float
    #     viewmatrix: torch.Tensor
    #     projmatrix: torch.Tensor
    #     sh_degree: int
    #     campos: torch.Tensor
    #     prefiltered: bool
    #     debug: bool = False
    # ddgr.GaussianRasterizationSettings = PatchedGaussianRasterizationSettings

    # disable get_has_ddp_rank print in `BaseInpaintingTrainingModule`
    os.environ["NODE_RANK"] = "0"

    from thirdparty.pano2room.modules.inpainters.lama.saicinpainting.training.trainers import (
        load_checkpoint,
    )
    from thirdparty.pano2room.modules.inpainters.lama_inpainter import (
        LamaInpainter,
    )

    def patched_lama_inpaint_init(self):
        zip_path = hf_hub_download(
            repo_id="smartywu/big-lama",
            filename="big-lama.zip",
            repo_type="model",
        )
        extract_dir = os.path.splitext(zip_path)[0]

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        config_path = os.path.join(extract_dir, 'big-lama', 'config.yaml')
        checkpoint_path = os.path.join(
            extract_dir, 'big-lama/models/best.ckpt'
        )
        train_config = OmegaConf.load(config_path)
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        self.model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu'
        )
        self.model.freeze()

    LamaInpainter.__init__ = patched_lama_inpaint_init

    from diffusers import StableDiffusionInpaintPipeline
    from thirdparty.pano2room.modules.inpainters.SDFT_inpainter import (
        SDFTInpainter,
    )

    def patched_sd_inpaint_init(self, subset_name=None):
        super(SDFTInpainter, self).__init__()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.enable_model_cpu_offload()
        self.inpaint_pipe = pipe

    SDFTInpainter.__init__ = patched_sd_inpaint_init


def monkey_patch_maniskill():
    from mani_skill.envs.scene import ManiSkillScene

    def get_sensor_images(
        self, obs: dict[str, any]
    ) -> dict[str, dict[str, torch.Tensor]]:
        sensor_data = dict()
        for name, sensor in self.sensors.items():
            sensor_data[name] = sensor.get_images(obs[name])
        return sensor_data

    def get_human_render_camera_images(
        self, camera_name: str = None, return_alpha: bool = False
    ) -> dict[str, torch.Tensor]:
        def get_rgba_tensor(camera, return_alpha):
            color = camera.get_obs(
                rgb=True, depth=False, segmentation=False, position=False
            )["rgb"]
            if return_alpha:
                seg_labels = camera.get_obs(
                    rgb=False, depth=False, segmentation=True, position=False
                )["segmentation"]
                masks = np.where((seg_labels.cpu() > 0), 255, 0).astype(
                    np.uint8
                )
                masks = torch.tensor(masks).to(color.device)
                color = torch.concat([color, masks], dim=-1)

            return color

        image_data = dict()
        if self.gpu_sim_enabled:
            if self.parallel_in_single_scene:
                for name, camera in self.human_render_cameras.items():
                    camera.camera._render_cameras[0].take_picture()
                    rgba = get_rgba_tensor(camera, return_alpha)
                    image_data[name] = rgba
            else:
                for name, camera in self.human_render_cameras.items():
                    if camera_name is not None and name != camera_name:
                        continue
                    assert camera.config.shader_config.shader_pack not in [
                        "rt",
                        "rt-fast",
                        "rt-med",
                    ], "ray tracing shaders do not work with parallel rendering"
                    camera.capture()
                    rgba = get_rgba_tensor(camera, return_alpha)
                    image_data[name] = rgba
        else:
            for name, camera in self.human_render_cameras.items():
                if camera_name is not None and name != camera_name:
                    continue
                camera.capture()
                rgba = get_rgba_tensor(camera, return_alpha)
                image_data[name] = rgba

        return image_data

    ManiSkillScene.get_sensor_images = get_sensor_images
    ManiSkillScene.get_human_render_camera_images = (
        get_human_render_camera_images
    )
