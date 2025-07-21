import logging
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from shutil import copy, rmtree

import torch
import tyro
from huggingface_hub import snapshot_download
from packaging import version

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# TorchVision monkey patch for >0.16
if version.parse(torch.__version__) >= version.parse("0.16"):
    import sys
    import types

    import torchvision.transforms.functional as TF

    functional_tensor = types.ModuleType(
        "torchvision.transforms.functional_tensor"
    )
    functional_tensor.rgb_to_grayscale = TF.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from gsplat.distributed import cli
from txt2panoimg import Text2360PanoramaImagePipeline
from embodied_gen.trainer.gsplat_trainer import (
    DefaultStrategy,
    GsplatTrainConfig,
)
from embodied_gen.trainer.gsplat_trainer import entrypoint as gsplat_entrypoint
from embodied_gen.trainer.pono2mesh_trainer import Pano2MeshSRPipeline
from embodied_gen.utils.config import Pano2MeshSRConfig
from embodied_gen.utils.gaussian import restore_scene_scale_and_position
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import is_image_file, parse_text_prompts
from embodied_gen.validators.quality_checkers import (
    PanoHeightEstimator,
    PanoImageOccChecker,
)

__all__ = [
    "generate_pano_image",
    "entrypoint",
]


@dataclass
class Scene3DGenConfig:
    prompts: list[str]  # Text desc of indoor room or style reference image.
    output_dir: str
    seed: int | None = None
    real_height: float | None = None  # The real height of the room in meters.
    pano_image_only: bool = False
    disable_pano_check: bool = False
    keep_middle_result: bool = False
    n_retry: int = 7
    gs3d: GsplatTrainConfig = field(
        default_factory=lambda: GsplatTrainConfig(
            strategy=DefaultStrategy(verbose=True),
            max_steps=4000,
            init_opa=0.9,
            opacity_reg=2e-3,
            sh_degree=0,
            means_lr=1e-4,
            scales_lr=1e-3,
        )
    )


def generate_pano_image(
    prompt: str,
    output_path: str,
    pipeline,
    seed: int,
    n_retry: int,
    checker=None,
    num_inference_steps: int = 40,
) -> None:
    for i in range(n_retry):
        logger.info(
            f"GEN Panorama: Retry {i+1}/{n_retry} for prompt: {prompt}, seed: {seed}"
        )
        if is_image_file(prompt):
            raise NotImplementedError("Image mode not implemented yet.")
        else:
            txt_prompt = f"{prompt}, spacious, empty, wide open, open floor, minimal furniture"
            inputs = {
                "prompt": txt_prompt,
                "num_inference_steps": num_inference_steps,
                "upscale": False,
                "seed": seed,
            }
            pano_image = pipeline(inputs)

        pano_image.save(output_path)
        if checker is None:
            break

        flag, response = checker(pano_image)
        logger.warning(f"{response}, image saved in {output_path}")
        if flag is True or flag is None:
            break

        seed = random.randint(0, 100000)

    return


def entrypoint(*args, **kwargs):
    cfg = tyro.cli(Scene3DGenConfig)

    # Init global models.
    model_path = snapshot_download("archerfmy0831/sd-t2i-360panoimage")
    IMG2PANO_PIPE = Text2360PanoramaImagePipeline(
        model_path, torch_dtype=torch.float16, device="cuda"
    )
    PANOMESH_CFG = Pano2MeshSRConfig()
    PANO2MESH_PIPE = Pano2MeshSRPipeline(PANOMESH_CFG)
    PANO_CHECKER = PanoImageOccChecker(GPT_CLIENT, box_hw=[95, 1000])
    PANOHEIGHT_ESTOR = PanoHeightEstimator(GPT_CLIENT)

    prompts = parse_text_prompts(cfg.prompts)
    for idx, prompt in enumerate(prompts):
        start_time = time.time()
        output_dir = os.path.join(cfg.output_dir, f"scene_{idx:04d}")
        os.makedirs(output_dir, exist_ok=True)
        pano_path = os.path.join(output_dir, "pano_image.png")
        with open(f"{output_dir}/prompt.txt", "w") as f:
            f.write(prompt)

        generate_pano_image(
            prompt,
            pano_path,
            IMG2PANO_PIPE,
            cfg.seed if cfg.seed is not None else random.randint(0, 100000),
            cfg.n_retry,
            checker=None if cfg.disable_pano_check else PANO_CHECKER,
        )

        if cfg.pano_image_only:
            continue

        logger.info("GEN and REPAIR Mesh from Panorama...")
        PANO2MESH_PIPE(pano_path, output_dir)

        logger.info("TRAIN 3DGS from Mesh Init and Cube Image...")
        cfg.gs3d.data_dir = output_dir
        cfg.gs3d.result_dir = f"{output_dir}/gaussian"
        cfg.gs3d.adjust_steps(cfg.gs3d.steps_scaler)
        torch.set_default_device("cpu")  # recover default setting.
        cli(gsplat_entrypoint, cfg.gs3d, verbose=True)

        # Clean up the middle results.
        gs_path = (
            f"{cfg.gs3d.result_dir}/ply/point_cloud_{cfg.gs3d.max_steps-1}.ply"
        )
        copy(gs_path, f"{output_dir}/gs_model.ply")
        video_path = f"{cfg.gs3d.result_dir}/renders/video_step{cfg.gs3d.max_steps-1}.mp4"
        copy(video_path, f"{output_dir}/video.mp4")
        gs_cfg_path = f"{cfg.gs3d.result_dir}/cfg.yml"
        copy(gs_cfg_path, f"{output_dir}/gsplat_cfg.yml")
        if not cfg.keep_middle_result:
            rmtree(cfg.gs3d.result_dir, ignore_errors=True)
            os.remove(f"{output_dir}/{PANOMESH_CFG.gs_data_file}")

        real_height = (
            PANOHEIGHT_ESTOR(pano_path)
            if cfg.real_height is None
            else cfg.real_height
        )
        gs_path = os.path.join(output_dir, "gs_model.ply")
        mesh_path = os.path.join(output_dir, "mesh_model.ply")
        restore_scene_scale_and_position(real_height, mesh_path, gs_path)

        elapsed_time = (time.time() - start_time) / 60
        logger.info(
            f"FINISHED 3D scene generation in {output_dir} in {elapsed_time:.2f} mins."
        )


if __name__ == "__main__":
    entrypoint()
