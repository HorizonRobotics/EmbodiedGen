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

__all__ = [
    "monkey_patch_pano2room",
    "monkey_patch_maniskill",
    "monkey_patch_sam3d",
]


def monkey_path_trellis():
    import torch.nn.functional as F

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    sys.path.append(os.path.join(current_dir, "../.."))

    from thirdparty.TRELLIS.trellis.representations import Gaussian
    from thirdparty.TRELLIS.trellis.representations.gaussian.general_utils import (
        build_scaling_rotation,
        inverse_sigmoid,
        strip_symmetric,
    )

    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.expanduser(
        "~/.cache/torch_extensions"
    )
    os.environ["SPCONV_ALGO"] = "auto"  # Can be 'native' or 'auto'
    os.environ['ATTN_BACKEND'] = (
        "xformers"  # Can be 'flash-attn' or 'xformers'
    )
    from thirdparty.TRELLIS.trellis.modules.sparse import set_attn

    set_attn("xformers")

    def patched_setup_functions(self):
        def inverse_softplus(x):
            return x + torch.log(-torch.expm1(-x))

        def build_covariance_from_scaling_rotation(
            scaling, scaling_modifier, rotation
        ):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = F.softplus
            self.inverse_scaling_activation = inverse_softplus

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = F.normalize

        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias)
        ).to(self.device)
        self.rots_bias = torch.zeros((4)).to(self.device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias)
        ).to(self.device)

    Gaussian.setup_functions = patched_setup_functions


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
                masks = np.where((seg_labels.cpu() > 1), 255, 0).astype(
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


def monkey_patch_sam3d():
    from typing import Optional, Union

    from embodied_gen.data.utils import model_device_ctx
    from embodied_gen.utils.log import logger

    os.environ["LIDRA_SKIP_INIT"] = "true"

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    sam3d_root = os.path.abspath(
        os.path.join(current_dir, "../../thirdparty/sam3d")
    )
    if sam3d_root not in sys.path:
        sys.path.insert(0, sam3d_root)

    def patch_pointmap_infer_pipeline():
        from copy import deepcopy

        try:
            from sam3d_objects.pipeline.inference_pipeline_pointmap import (
                InferencePipelinePointMap,
            )
        except ImportError:
            logger.error(
                "[MonkeyPatch]: Could not import sam3d_objects directly. Check paths."
            )
            return

        def patch_run(
            self,
            image: Union[None, Image.Image, np.ndarray],
            mask: Union[None, Image.Image, np.ndarray] = None,
            seed: Optional[int] = None,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=True,
            use_vertex_color=False,
            stage1_inference_steps=None,
            stage2_inference_steps=None,
            use_stage1_distillation=False,
            use_stage2_distillation=False,
            pointmap=None,
            decode_formats=None,
            estimate_plane=False,
        ) -> dict:
            image = self.merge_image_and_mask(image, mask)
            with self.device:
                pointmap_dict = self.compute_pointmap(image, pointmap)
                pointmap = pointmap_dict["pointmap"]
                pts = type(self)._down_sample_img(pointmap)
                pts_colors = type(self)._down_sample_img(
                    pointmap_dict["pts_color"]
                )

                if estimate_plane:
                    return self.estimate_plane(pointmap_dict, image)

                ss_input_dict = self.preprocess_image(
                    image, self.ss_preprocessor, pointmap=pointmap
                )

                slat_input_dict = self.preprocess_image(
                    image, self.slat_preprocessor
                )
                if seed is not None:
                    torch.manual_seed(seed)

                with model_device_ctx(
                    self.models["ss_generator"],
                    self.models["ss_decoder"],
                    self.condition_embedders["ss_condition_embedder"],
                ):
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict,
                        inference_steps=stage1_inference_steps,
                        use_distillation=use_stage1_distillation,
                    )

                # We could probably use the decoder from the models themselves
                pointmap_scale = ss_input_dict.get("pointmap_scale", None)
                pointmap_shift = ss_input_dict.get("pointmap_shift", None)
                ss_return_dict.update(
                    self.pose_decoder(
                        ss_return_dict,
                        scene_scale=pointmap_scale,
                        scene_shift=pointmap_shift,
                    )
                )

                ss_return_dict["scale"] = (
                    ss_return_dict["scale"]
                    * ss_return_dict["downsample_factor"]
                )

                if stage1_only:
                    logger.info("Finished!")
                    ss_return_dict["voxel"] = (
                        ss_return_dict["coords"][:, 1:] / 64 - 0.5
                    )
                    return {
                        **ss_return_dict,
                        "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                        "pointmap_colors": pts_colors.cpu().permute(
                            (1, 2, 0)
                        ),  # HxWx3
                    }
                    # return ss_return_dict

                coords = ss_return_dict["coords"]
                with model_device_ctx(
                    self.models["slat_generator"],
                    self.condition_embedders["slat_condition_embedder"],
                ):
                    slat = self.sample_slat(
                        slat_input_dict,
                        coords,
                        inference_steps=stage2_inference_steps,
                        use_distillation=use_stage2_distillation,
                    )

                with model_device_ctx(
                    self.models["slat_decoder_mesh"],
                    self.models["slat_decoder_gs"],
                    self.models["slat_decoder_gs_4"],
                ):
                    outputs = self.decode_slat(
                        slat,
                        (
                            self.decode_formats
                            if decode_formats is None
                            else decode_formats
                        ),
                    )

                outputs = self.postprocess_slat_output(
                    outputs,
                    with_mesh_postprocess,
                    with_texture_baking,
                    use_vertex_color,
                )
                glb = outputs.get("glb", None)

                try:
                    if (
                        with_layout_postprocess
                        and self.layout_post_optimization_method is not None
                    ):
                        assert (
                            glb is not None
                        ), "require mesh to run postprocessing"
                        logger.info(
                            "Running layout post optimization method..."
                        )
                        postprocessed_pose = self.run_post_optimization(
                            deepcopy(glb),
                            pointmap_dict["intrinsics"],
                            ss_return_dict,
                            ss_input_dict,
                        )
                        ss_return_dict.update(postprocessed_pose)
                except Exception as e:
                    logger.error(
                        f"Error during layout post optimization: {e}",
                        exc_info=True,
                    )

                result = {
                    **ss_return_dict,
                    **outputs,
                    "pointmap": pts.cpu().permute((1, 2, 0)),
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
                }
                return result

        InferencePipelinePointMap.run = patch_run

    def patch_infer_init():
        import torch

        try:
            from sam3d_objects.pipeline import preprocess_utils
            from sam3d_objects.pipeline.inference_pipeline_pointmap import (
                InferencePipeline,
            )
            from sam3d_objects.pipeline.inference_utils import (
                SLAT_MEAN,
                SLAT_STD,
            )
        except ImportError:
            print(
                "[MonkeyPatch] Error: Could not import sam3d_objects directly for infer pipeline."
            )
            return

        def patch_init(
            self,
            ss_generator_config_path,
            ss_generator_ckpt_path,
            slat_generator_config_path,
            slat_generator_ckpt_path,
            ss_decoder_config_path,
            ss_decoder_ckpt_path,
            slat_decoder_gs_config_path,
            slat_decoder_gs_ckpt_path,
            slat_decoder_mesh_config_path,
            slat_decoder_mesh_ckpt_path,
            slat_decoder_gs_4_config_path=None,
            slat_decoder_gs_4_ckpt_path=None,
            ss_encoder_config_path=None,
            ss_encoder_ckpt_path=None,
            decode_formats=["gaussian", "mesh"],
            dtype="bfloat16",
            pad_size=1.0,
            version="v0",
            device="cuda",
            ss_preprocessor=preprocess_utils.get_default_preprocessor(),
            slat_preprocessor=preprocess_utils.get_default_preprocessor(),
            ss_condition_input_mapping=["image"],
            slat_condition_input_mapping=["image"],
            pose_decoder_name="default",
            workspace_dir="",
            downsample_ss_dist=0,  # the distance we use to downsample
            ss_inference_steps=25,
            ss_rescale_t=3,
            ss_cfg_strength=7,
            ss_cfg_interval=[0, 500],
            ss_cfg_strength_pm=0.0,
            slat_inference_steps=25,
            slat_rescale_t=3,
            slat_cfg_strength=5,
            slat_cfg_interval=[0, 500],
            rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d,
            shape_model_dtype=None,
            compile_model=False,
            slat_mean=SLAT_MEAN,
            slat_std=SLAT_STD,
        ):
            self.rendering_engine = rendering_engine
            self.device = torch.device(device)
            self.compile_model = compile_model
            with self.device:
                self.decode_formats = decode_formats
                self.pad_size = pad_size
                self.version = version
                self.ss_condition_input_mapping = ss_condition_input_mapping
                self.slat_condition_input_mapping = (
                    slat_condition_input_mapping
                )
                self.workspace_dir = workspace_dir
                self.downsample_ss_dist = downsample_ss_dist
                self.ss_inference_steps = ss_inference_steps
                self.ss_rescale_t = ss_rescale_t
                self.ss_cfg_strength = ss_cfg_strength
                self.ss_cfg_interval = ss_cfg_interval
                self.ss_cfg_strength_pm = ss_cfg_strength_pm
                self.slat_inference_steps = slat_inference_steps
                self.slat_rescale_t = slat_rescale_t
                self.slat_cfg_strength = slat_cfg_strength
                self.slat_cfg_interval = slat_cfg_interval

                self.dtype = self._get_dtype(dtype)
                if shape_model_dtype is None:
                    self.shape_model_dtype = self.dtype
                else:
                    self.shape_model_dtype = self._get_dtype(shape_model_dtype)

                # Setup preprocessors
                self.pose_decoder = self.init_pose_decoder(
                    ss_generator_config_path, pose_decoder_name
                )
                self.ss_preprocessor = self.init_ss_preprocessor(
                    ss_preprocessor, ss_generator_config_path
                )
                self.slat_preprocessor = slat_preprocessor

                raw_device = self.device
                self.device = torch.device("cpu")
                ss_generator = self.init_ss_generator(
                    ss_generator_config_path, ss_generator_ckpt_path
                )
                slat_generator = self.init_slat_generator(
                    slat_generator_config_path, slat_generator_ckpt_path
                )
                ss_decoder = self.init_ss_decoder(
                    ss_decoder_config_path, ss_decoder_ckpt_path
                )
                ss_encoder = self.init_ss_encoder(
                    ss_encoder_config_path, ss_encoder_ckpt_path
                )
                slat_decoder_gs = self.init_slat_decoder_gs(
                    slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
                )
                slat_decoder_gs_4 = self.init_slat_decoder_gs(
                    slat_decoder_gs_4_config_path, slat_decoder_gs_4_ckpt_path
                )
                slat_decoder_mesh = self.init_slat_decoder_mesh(
                    slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
                )

                # Load conditioner embedder so that we only load it once
                ss_condition_embedder = self.init_ss_condition_embedder(
                    ss_generator_config_path, ss_generator_ckpt_path
                )
                slat_condition_embedder = self.init_slat_condition_embedder(
                    slat_generator_config_path, slat_generator_ckpt_path
                )
                self.device = raw_device

                self.condition_embedders = {
                    "ss_condition_embedder": ss_condition_embedder,
                    "slat_condition_embedder": slat_condition_embedder,
                }

                # override generator and condition embedder setting
                self.override_ss_generator_cfg_config(
                    ss_generator,
                    cfg_strength=ss_cfg_strength,
                    inference_steps=ss_inference_steps,
                    rescale_t=ss_rescale_t,
                    cfg_interval=ss_cfg_interval,
                    cfg_strength_pm=ss_cfg_strength_pm,
                )
                self.override_slat_generator_cfg_config(
                    slat_generator,
                    cfg_strength=slat_cfg_strength,
                    inference_steps=slat_inference_steps,
                    rescale_t=slat_rescale_t,
                    cfg_interval=slat_cfg_interval,
                )

                self.models = torch.nn.ModuleDict(
                    {
                        "ss_generator": ss_generator,
                        "slat_generator": slat_generator,
                        "ss_encoder": ss_encoder,
                        "ss_decoder": ss_decoder,
                        "slat_decoder_gs": slat_decoder_gs,
                        "slat_decoder_gs_4": slat_decoder_gs_4,
                        "slat_decoder_mesh": slat_decoder_mesh,
                    }
                )
                logger.info("Loading SAM3D model weights completed.")

                if self.compile_model:
                    logger.info("Compiling model...")
                    self._compile()
                    logger.info("Model compilation completed!")
                self.slat_mean = torch.tensor(slat_mean)
                self.slat_std = torch.tensor(slat_std)

        InferencePipeline.__init__ = patch_init

    patch_pointmap_infer_pipeline()
    patch_infer_init()

    return
