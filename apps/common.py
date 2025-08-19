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

import gc
import logging
import os
import shutil
import subprocess
import sys
from glob import glob

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F
import trimesh
from easydict import EasyDict as edict
from gradio.themes import Soft
from gradio.themes.utils.colors import gray, neutral, slate, stone, teal, zinc
from PIL import Image
from embodied_gen.data.backproject_v2 import entrypoint as backproject_api
from embodied_gen.data.differentiable_render import entrypoint as render_api
from embodied_gen.data.utils import trellis_preprocess, zip_files
from embodied_gen.models.delight_model import DelightingModel
from embodied_gen.models.gs_model import GaussianOperator
from embodied_gen.models.segment_model import (
    BMGG14Remover,
    RembgRemover,
    SAMPredictor,
)
from embodied_gen.models.sr_model import ImageRealESRGAN, ImageStableSR
from embodied_gen.scripts.render_gs import entrypoint as render_gs_api
from embodied_gen.scripts.render_mv import build_texture_gen_pipe, infer_pipe
from embodied_gen.scripts.text2image import (
    build_text2img_ip_pipeline,
    build_text2img_pipeline,
    text2img_gen,
)
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.process_media import (
    filter_image_small_connected_components,
    merge_images_video,
)
from embodied_gen.utils.tags import VERSION
from embodied_gen.utils.trender import render_video
from embodied_gen.validators.quality_checkers import (
    BaseChecker,
    ImageAestheticChecker,
    ImageSegChecker,
    MeshGeoChecker,
)
from embodied_gen.validators.urdf_convertor import URDFGenerator

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, ".."))
from thirdparty.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from thirdparty.TRELLIS.trellis.representations import (
    Gaussian,
    MeshExtractResult,
)
from thirdparty.TRELLIS.trellis.representations.gaussian.general_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from thirdparty.TRELLIS.trellis.utils import postprocessing_utils

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


os.environ["TORCH_EXTENSIONS_DIR"] = os.path.expanduser(
    "~/.cache/torch_extensions"
)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["SPCONV_ALGO"] = "native"

MAX_SEED = 100000


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


DELIGHT = DelightingModel()
IMAGESR_MODEL = ImageRealESRGAN(outscale=4)
# IMAGESR_MODEL = ImageStableSR()
if os.getenv("GRADIO_APP") == "imageto3d":
    RBG_REMOVER = RembgRemover()
    RBG14_REMOVER = BMGG14Remover()
    SAM_PREDICTOR = SAMPredictor(model_type="vit_h", device="cpu")
    PIPELINE = TrellisImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS-image-large"
    )
    # PIPELINE.cuda()
    SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
    GEO_CHECKER = MeshGeoChecker(GPT_CLIENT)
    AESTHETIC_CHECKER = ImageAestheticChecker()
    CHECKERS = [GEO_CHECKER, SEG_CHECKER, AESTHETIC_CHECKER]
    TMP_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sessions/imageto3d"
    )
elif os.getenv("GRADIO_APP") == "textto3d":
    RBG_REMOVER = RembgRemover()
    RBG14_REMOVER = BMGG14Remover()
    PIPELINE = TrellisImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS-image-large"
    )
    # PIPELINE.cuda()
    text_model_dir = "weights/Kolors"
    PIPELINE_IMG_IP = build_text2img_ip_pipeline(text_model_dir, ref_scale=0.3)
    PIPELINE_IMG = build_text2img_pipeline(text_model_dir)
    SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
    GEO_CHECKER = MeshGeoChecker(GPT_CLIENT)
    AESTHETIC_CHECKER = ImageAestheticChecker()
    CHECKERS = [GEO_CHECKER, SEG_CHECKER, AESTHETIC_CHECKER]
    TMP_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sessions/textto3d"
    )
elif os.getenv("GRADIO_APP") == "texture_edit":
    PIPELINE_IP = build_texture_gen_pipe(
        base_ckpt_dir="./weights",
        ip_adapt_scale=0.7,
        device="cuda",
    )
    PIPELINE = build_texture_gen_pipe(
        base_ckpt_dir="./weights",
        ip_adapt_scale=0,
        device="cuda",
    )
    TMP_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sessions/texture_edit"
    )

os.makedirs(TMP_DIR, exist_ok=True)


lighting_css = """
<style>
#lighter_mesh canvas {
    filter: brightness(1.9) !important;
}
</style>
"""

image_css = """
<style>
.image_fit .image-frame {
object-fit: contain !important;
height: 100% !important;
}
</style>
"""

custom_theme = Soft(
    primary_hue=stone,
    secondary_hue=gray,
    radius_size="md",
    text_size="sm",
    spacing_size="sm",
)


def start_session(req: gr.Request) -> None:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request) -> None:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


@spaces.GPU
def preprocess_image_fn(
    image: str | np.ndarray | Image.Image, rmbg_tag: str = "rembg"
) -> tuple[Image.Image, Image.Image]:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_cache = image.copy().resize((512, 512))

    bg_remover = RBG_REMOVER if rmbg_tag == "rembg" else RBG14_REMOVER
    image = bg_remover(image)
    image = trellis_preprocess(image)

    return image, image_cache


def preprocess_sam_image_fn(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    sam_image = SAM_PREDICTOR.preprocess_image(image)
    image_cache = Image.fromarray(sam_image).resize((512, 512))
    SAM_PREDICTOR.predictor.set_image(sam_image)

    return sam_image, image_cache


def active_btn_by_content(content: gr.Image) -> gr.Button:
    interactive = True if content is not None else False

    return gr.Button(interactive=interactive)


def active_btn_by_text_content(content: gr.Textbox) -> gr.Button:
    if content is not None and len(content) > 0:
        interactive = True
    else:
        interactive = False

    return gr.Button(interactive=interactive)


def get_selected_image(
    choice: str, sample1: str, sample2: str, sample3: str
) -> str:
    if choice == "sample1":
        return sample1
    elif choice == "sample2":
        return sample2
    elif choice == "sample3":
        return sample3
    else:
        raise ValueError(f"Invalid choice: {choice}")


def get_cached_image(image_path: str) -> Image.Image:
    if isinstance(image_path, Image.Image):
        return image_path
    return Image.open(image_path).resize((512, 512))


@spaces.GPU
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict, device: str = "cpu") -> tuple[Gaussian, dict]:
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
        device=device,
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device=device)
    gs._features_dc = torch.tensor(
        state["gaussian"]["_features_dc"], device=device
    )
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device=device)
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device=device)
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device=device)

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device=device),
        faces=torch.tensor(state["mesh"]["faces"], device=device),
    )

    return gs, mesh


def get_seed(randomize_seed: bool, seed: int, max_seed: int = MAX_SEED) -> int:
    return np.random.randint(0, max_seed) if randomize_seed else seed


def select_point(
    image: np.ndarray,
    sel_pix: list,
    point_type: str,
    evt: gr.SelectData,
):
    if point_type == "foreground_point":
        sel_pix.append((evt.index, 1))  # append the foreground_point
    elif point_type == "background_point":
        sel_pix.append((evt.index, 0))  # append the background_point
    else:
        sel_pix.append((evt.index, 1))  # default foreground_point

    masks = SAM_PREDICTOR.generate_masks(image, sel_pix)
    seg_image = SAM_PREDICTOR.get_segmented_image(image, masks)

    for point, label in sel_pix:
        color = (255, 0, 0) if label == 0 else (0, 255, 0)
        marker_type = 1 if label == 0 else 5
        cv2.drawMarker(
            image,
            point,
            color,
            markerType=marker_type,
            markerSize=15,
            thickness=10,
        )

    torch.cuda.empty_cache()

    return (image, masks), seg_image


@spaces.GPU
def image_to_3d(
    image: Image.Image,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    raw_image_cache: Image.Image,
    sam_image: Image.Image = None,
    is_sam_image: bool = False,
    req: gr.Request = None,
) -> tuple[dict, str]:
    if is_sam_image:
        seg_image = filter_image_small_connected_components(sam_image)
        seg_image = Image.fromarray(seg_image, mode="RGBA")
        seg_image = trellis_preprocess(seg_image)
    else:
        seg_image = image

    if isinstance(seg_image, np.ndarray):
        seg_image = Image.fromarray(seg_image)

    output_root = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(output_root, exist_ok=True)
    seg_image.save(f"{output_root}/seg_image.png")
    raw_image_cache.save(f"{output_root}/raw_image.png")
    PIPELINE.cuda()
    outputs = PIPELINE.run(
        seg_image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    # Set to cpu for memory saving.
    PIPELINE.cpu()

    gs_model = outputs["gaussian"][0]
    mesh_model = outputs["mesh"][0]
    color_images = render_video(gs_model)["color"]
    normal_images = render_video(mesh_model)["normal"]

    video_path = os.path.join(output_root, "gs_mesh.mp4")
    merge_images_video(color_images, normal_images, video_path)
    state = pack_state(gs_model, mesh_model)

    gc.collect()
    torch.cuda.empty_cache()

    return state, video_path


@spaces.GPU
def extract_3d_representations(
    state: dict, enable_delight: bool, texture_size: int, req: gr.Request
):
    output_root = TMP_DIR
    output_root = os.path.join(output_root, str(req.session_hash))
    gs_model, mesh_model = unpack_state(state, device="cuda")

    mesh = postprocessing_utils.to_glb(
        gs_model,
        mesh_model,
        simplify=0.9,
        texture_size=1024,
        verbose=True,
    )
    filename = "sample"
    gs_path = os.path.join(output_root, f"{filename}_gs.ply")
    gs_model.save_ply(gs_path)

    # Rotate mesh and GS by 90 degrees around Z-axis.
    rot_matrix = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    # Addtional rotation for GS to align mesh.
    gs_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array(
        rot_matrix
    )
    pose = GaussianOperator.trans_to_quatpose(gs_rot)
    aligned_gs_path = gs_path.replace(".ply", "_aligned.ply")
    GaussianOperator.resave_ply(
        in_ply=gs_path,
        out_ply=aligned_gs_path,
        instance_pose=pose,
    )

    mesh.vertices = mesh.vertices @ np.array(rot_matrix)
    mesh_obj_path = os.path.join(output_root, f"{filename}.obj")
    mesh.export(mesh_obj_path)
    mesh_glb_path = os.path.join(output_root, f"{filename}.glb")
    mesh.export(mesh_glb_path)

    torch.cuda.empty_cache()

    return mesh_glb_path, gs_path, mesh_obj_path, aligned_gs_path


def extract_3d_representations_v2(
    state: dict,
    enable_delight: bool,
    texture_size: int,
    req: gr.Request,
):
    output_root = TMP_DIR
    user_dir = os.path.join(output_root, str(req.session_hash))
    gs_model, mesh_model = unpack_state(state, device="cpu")

    filename = "sample"
    gs_path = os.path.join(user_dir, f"{filename}_gs.ply")
    gs_model.save_ply(gs_path)

    # Rotate mesh and GS by 90 degrees around Z-axis.
    rot_matrix = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    gs_add_rot = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    mesh_add_rot = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    # Addtional rotation for GS to align mesh.
    gs_rot = np.array(gs_add_rot) @ np.array(rot_matrix)
    pose = GaussianOperator.trans_to_quatpose(gs_rot)
    aligned_gs_path = gs_path.replace(".ply", "_aligned.ply")
    GaussianOperator.resave_ply(
        in_ply=gs_path,
        out_ply=aligned_gs_path,
        instance_pose=pose,
        device="cpu",
    )
    color_path = os.path.join(user_dir, "color.png")
    render_gs_api(aligned_gs_path, color_path)

    mesh = trimesh.Trimesh(
        vertices=mesh_model.vertices.cpu().numpy(),
        faces=mesh_model.faces.cpu().numpy(),
    )
    mesh.vertices = mesh.vertices @ np.array(mesh_add_rot)
    mesh.vertices = mesh.vertices @ np.array(rot_matrix)

    mesh_obj_path = os.path.join(user_dir, f"{filename}.obj")
    mesh.export(mesh_obj_path)

    mesh = backproject_api(
        delight_model=DELIGHT,
        imagesr_model=IMAGESR_MODEL,
        color_path=color_path,
        mesh_path=mesh_obj_path,
        output_path=mesh_obj_path,
        skip_fix_mesh=False,
        delight=enable_delight,
        texture_wh=[texture_size, texture_size],
    )

    mesh_glb_path = os.path.join(user_dir, f"{filename}.glb")
    mesh.export(mesh_glb_path)

    return mesh_glb_path, gs_path, mesh_obj_path, aligned_gs_path


def extract_urdf(
    gs_path: str,
    mesh_obj_path: str,
    asset_cat_text: str,
    height_range_text: str,
    mass_range_text: str,
    asset_version_text: str,
    req: gr.Request = None,
):
    output_root = TMP_DIR
    if req is not None:
        output_root = os.path.join(output_root, str(req.session_hash))

    # Convert to URDF and recover attrs by GPT.
    filename = "sample"
    urdf_convertor = URDFGenerator(GPT_CLIENT, render_view_num=4)
    asset_attrs = {
        "version": VERSION,
        "gs_model": f"{urdf_convertor.output_mesh_dir}/{filename}_gs.ply",
    }
    if asset_version_text:
        asset_attrs["version"] = asset_version_text
    if asset_cat_text:
        asset_attrs["category"] = asset_cat_text.lower()
    if height_range_text:
        try:
            min_height, max_height = map(float, height_range_text.split("-"))
            asset_attrs["min_height"] = min_height
            asset_attrs["max_height"] = max_height
        except ValueError:
            return "Invalid height input format. Use the format: min-max."
    if mass_range_text:
        try:
            min_mass, max_mass = map(float, mass_range_text.split("-"))
            asset_attrs["min_mass"] = min_mass
            asset_attrs["max_mass"] = max_mass
        except ValueError:
            return "Invalid mass input format. Use the format: min-max."

    urdf_path = urdf_convertor(
        mesh_path=mesh_obj_path,
        output_root=f"{output_root}/URDF_{filename}",
        **asset_attrs,
    )

    # Rescale GS and save to URDF/mesh folder.
    real_height = urdf_convertor.get_attr_from_urdf(
        urdf_path, attr_name="real_height"
    )
    out_gs = f"{output_root}/URDF_{filename}/{urdf_convertor.output_mesh_dir}/{filename}_gs.ply"  # noqa
    GaussianOperator.resave_ply(
        in_ply=gs_path,
        out_ply=out_gs,
        real_height=real_height,
        device="cpu",
    )

    # Quality check and update .urdf file.
    mesh_out = f"{output_root}/URDF_{filename}/{urdf_convertor.output_mesh_dir}/{filename}.obj"  # noqa
    trimesh.load(mesh_out).export(mesh_out.replace(".obj", ".glb"))
    # image_paths = render_asset3d(
    #     mesh_path=mesh_out,
    #     output_root=f"{output_root}/URDF_{filename}",
    #     output_subdir="qa_renders",
    #     num_images=8,
    #     elevation=(30, -30),
    #     distance=5.5,
    # )

    image_dir = f"{output_root}/URDF_{filename}/{urdf_convertor.output_render_dir}/image_color"  # noqa
    image_paths = glob(f"{image_dir}/*.png")
    images_list = []
    for checker in CHECKERS:
        images = image_paths
        if isinstance(checker, ImageSegChecker):
            images = [
                f"{TMP_DIR}/{req.session_hash}/raw_image.png",
                f"{TMP_DIR}/{req.session_hash}/seg_image.png",
            ]
        images_list.append(images)

    results = BaseChecker.validate(CHECKERS, images_list)
    urdf_convertor.add_quality_tag(urdf_path, results)

    # Zip urdf files
    urdf_zip = zip_files(
        input_paths=[
            f"{output_root}/URDF_{filename}/{urdf_convertor.output_mesh_dir}",
            f"{output_root}/URDF_{filename}/{filename}.urdf",
        ],
        output_zip=f"{output_root}/urdf_{filename}.zip",
    )

    estimated_type = urdf_convertor.estimated_attrs["category"]
    estimated_height = urdf_convertor.estimated_attrs["height"]
    estimated_mass = urdf_convertor.estimated_attrs["mass"]
    estimated_mu = urdf_convertor.estimated_attrs["mu"]

    return (
        urdf_zip,
        estimated_type,
        estimated_height,
        estimated_mass,
        estimated_mu,
    )


@spaces.GPU
def text2image_fn(
    prompt: str,
    guidance_scale: float,
    infer_step: int = 50,
    ip_image: Image.Image | str = None,
    ip_adapt_scale: float = 0.3,
    image_wh: int | tuple[int, int] = [1024, 1024],
    rmbg_tag: str = "rembg",
    seed: int = None,
    n_sample: int = 3,
    req: gr.Request = None,
):
    if isinstance(image_wh, int):
        image_wh = (image_wh, image_wh)
    output_root = TMP_DIR
    if req is not None:
        output_root = os.path.join(output_root, str(req.session_hash))
        os.makedirs(output_root, exist_ok=True)

    pipeline = PIPELINE_IMG if ip_image is None else PIPELINE_IMG_IP
    if ip_image is not None:
        pipeline.set_ip_adapter_scale([ip_adapt_scale])

    images = text2img_gen(
        prompt=prompt,
        n_sample=n_sample,
        guidance_scale=guidance_scale,
        pipeline=pipeline,
        ip_image=ip_image,
        image_wh=image_wh,
        infer_step=infer_step,
        seed=seed,
    )

    for idx in range(len(images)):
        image = images[idx]
        images[idx], _ = preprocess_image_fn(image, rmbg_tag)

    save_paths = []
    for idx, image in enumerate(images):
        save_path = f"{output_root}/sample_{idx}.png"
        image.save(save_path)
        save_paths.append(save_path)

    logger.info(f"Images saved to {output_root}")

    gc.collect()
    torch.cuda.empty_cache()

    return save_paths + save_paths


@spaces.GPU
def generate_condition(mesh_path: str, req: gr.Request, uuid: str = "sample"):
    output_root = os.path.join(TMP_DIR, str(req.session_hash))

    _ = render_api(
        mesh_path=mesh_path,
        output_root=f"{output_root}/condition",
        uuid=str(uuid),
    )

    gc.collect()
    torch.cuda.empty_cache()

    return None, None, None


@spaces.GPU
def generate_texture_mvimages(
    prompt: str,
    controlnet_cond_scale: float = 0.55,
    guidance_scale: float = 9,
    strength: float = 0.9,
    num_inference_steps: int = 50,
    seed: int = 0,
    ip_adapt_scale: float = 0,
    ip_img_path: str = None,
    uid: str = "sample",
    sub_idxs: tuple[tuple[int]] = ((0, 1, 2), (3, 4, 5)),
    req: gr.Request = None,
) -> list[str]:
    output_root = os.path.join(TMP_DIR, str(req.session_hash))
    use_ip_adapter = True if ip_img_path and ip_adapt_scale > 0 else False
    PIPELINE_IP.set_ip_adapter_scale([ip_adapt_scale])
    img_save_paths = infer_pipe(
        index_file=f"{output_root}/condition/index.json",
        controlnet_cond_scale=controlnet_cond_scale,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        ip_adapt_scale=ip_adapt_scale,
        ip_img_path=ip_img_path,
        uid=uid,
        prompt=prompt,
        save_dir=f"{output_root}/multi_view",
        sub_idxs=sub_idxs,
        pipeline=PIPELINE_IP if use_ip_adapter else PIPELINE,
        seed=seed,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return img_save_paths + img_save_paths


def backproject_texture(
    mesh_path: str,
    input_image: str,
    texture_size: int,
    uuid: str = "sample",
    req: gr.Request = None,
) -> str:
    output_root = os.path.join(TMP_DIR, str(req.session_hash))
    output_dir = os.path.join(output_root, "texture_mesh")
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "backproject-cli",
        "--mesh_path",
        mesh_path,
        "--input_image",
        input_image,
        "--output_root",
        output_dir,
        "--uuid",
        f"{uuid}",
        "--texture_size",
        str(texture_size),
        "--skip_fix_mesh",
    ]

    _ = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8"
    )
    output_obj_mesh = os.path.join(output_dir, f"{uuid}.obj")
    output_glb_mesh = os.path.join(output_dir, f"{uuid}.glb")
    _ = trimesh.load(output_obj_mesh).export(output_glb_mesh)

    zip_file = zip_files(
        input_paths=[
            output_glb_mesh,
            output_obj_mesh,
            os.path.join(output_dir, "material.mtl"),
            os.path.join(output_dir, "material_0.png"),
        ],
        output_zip=os.path.join(output_dir, f"{uuid}.zip"),
    )

    gc.collect()
    torch.cuda.empty_cache()

    return output_glb_mesh, output_obj_mesh, zip_file


@spaces.GPU
def backproject_texture_v2(
    mesh_path: str,
    input_image: str,
    texture_size: int,
    enable_delight: bool = True,
    fix_mesh: bool = False,
    uuid: str = "sample",
    req: gr.Request = None,
) -> str:
    output_root = os.path.join(TMP_DIR, str(req.session_hash))
    output_dir = os.path.join(output_root, "texture_mesh")
    os.makedirs(output_dir, exist_ok=True)

    textured_mesh = backproject_api(
        delight_model=DELIGHT,
        imagesr_model=IMAGESR_MODEL,
        color_path=input_image,
        mesh_path=mesh_path,
        output_path=f"{output_dir}/{uuid}.obj",
        skip_fix_mesh=not fix_mesh,
        delight=enable_delight,
        texture_wh=[texture_size, texture_size],
    )

    output_obj_mesh = os.path.join(output_dir, f"{uuid}.obj")
    output_glb_mesh = os.path.join(output_dir, f"{uuid}.glb")
    _ = textured_mesh.export(output_glb_mesh)

    zip_file = zip_files(
        input_paths=[
            output_glb_mesh,
            output_obj_mesh,
            os.path.join(output_dir, "material.mtl"),
            os.path.join(output_dir, "material_0.png"),
        ],
        output_zip=os.path.join(output_dir, f"{uuid}.zip"),
    )

    gc.collect()
    torch.cuda.empty_cache()

    return output_glb_mesh, output_obj_mesh, zip_file


@spaces.GPU
def render_result_video(
    mesh_path: str, video_size: int, req: gr.Request, uuid: str = ""
) -> str:
    output_root = os.path.join(TMP_DIR, str(req.session_hash))
    output_dir = os.path.join(output_root, "texture_mesh")

    _ = render_api(
        mesh_path=mesh_path,
        output_root=output_dir,
        num_images=90,
        elevation=[20],
        with_mtl=True,
        pbr_light_factor=1,
        uuid=str(uuid),
        gen_color_mp4=True,
        gen_glonormal_mp4=True,
        distance=5.5,
        resolution_hw=(video_size, video_size),
    )

    gc.collect()
    torch.cuda.empty_cache()

    return f"{output_dir}/color.mp4"
