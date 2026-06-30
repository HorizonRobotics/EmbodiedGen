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
import random
from pathlib import Path
import tempfile

# GRADIO_APP == "imageto3d_sam3d", sam3d object model, by default.
# GRADIO_APP == "imageto3d", TRELLIS model.
#os.environ["GRADIO_APP"] = "imageto3d_sam3d"
os.environ["GRADIO_APP"] = "imageto3d"

# Keep Gradio temp/cache under project workspace to avoid /tmp permission issues.
_gradio_tmp_root = Path(__file__).resolve().parent.parent / "tmp" / "gradio"
_gradio_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_gradio_tmp_root))
os.environ.setdefault("TMPDIR", str(_gradio_tmp_root))
tempfile.tempdir = str(_gradio_tmp_root)
from glob import glob

import gradio as gr
import trimesh
from app_style import custom_theme, image_css, lighting_css
from embodied_gen.utils.tags import VERSION
from hunyuan_image3d_save import query_and_download_hunyuan_job
from hunyuan_image3d_submit import hunyuan_image3d_submit

app_name = os.getenv("GRADIO_APP")
if app_name == "imageto3d_sam3d":
    enable_pre_resize = False
    sample_step = 25
    bg_rm_model_name = "rembg"  # "rembg", "rmbg14"
elif app_name == "imageto3d":
    enable_pre_resize = True
    sample_step = 12
    bg_rm_model_name = "rembg"  # "rembg", "rmbg14"

MAX_SEED = 100000
SESSION_ROOT = Path("./tmp/gradio_sessions")
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


def start_session(req: gr.Request) -> None:
    session_hash = req.session_hash if req is not None else "default"
    (SESSION_ROOT / str(session_hash)).mkdir(parents=True, exist_ok=True)


def end_session(req: gr.Request) -> None:
    # Keep lightweight API mode side-effect free; no mandatory cleanup here.
    return None


def active_btn_by_content(content) -> gr.Button:
    return gr.Button(interactive=content is not None)


def get_seed(randomize_seed: bool, seed: int, max_seed: int = MAX_SEED) -> int:
    return random.randint(0, max_seed) if randomize_seed else seed


def preprocess_image_fn(image, rmbg_tag: str = "rembg", preprocess: bool = True):
    # API-only mode: avoid loading local segmentation/rembg models at startup.
    if image is None:
        return None, None
    image_cache = image.copy() if hasattr(image, "copy") else image
    return image, image_cache


def preprocess_sam_image_fn(image):
    # API-only mode: SAM model is not loaded; keep input passthrough.
    if image is None:
        return None, None
    image_cache = image.copy() if hasattr(image, "copy") else image
    return image, image_cache


def select_point(image, sel_pix, point_type, evt: gr.SelectData):
    # API-only mode: no SAM interaction; return current image unchanged.
    return (image, None), image


def extract_urdf_lazy(
    gs_path: str,
    mesh_obj_path: str,
    asset_cat_text: str,
    height_range_text: str,
    mass_range_text: str,
    asset_version_text: str,
    req: gr.Request = None,
):
    from common import extract_urdf as _extract_urdf
    if (not mesh_obj_path) and gs_path:
        session_hash = req.session_hash if req is not None else "default"
        obj_dir = Path("./tmp/gradio") / str(session_hash) / "hunyuan_image3d" / "obj"
        mesh_obj_path = convert_glb_to_obj(gs_path, str(obj_dir))

    return _extract_urdf(
        gs_path,
        mesh_obj_path,
        asset_cat_text,
        height_range_text,
        mass_range_text,
        asset_version_text,
        req,
    )


def image_to_3d_via_hunyuan_api(
    image,
    raw_image_cache,
    seed,
    req: gr.Request = None,
):
    if image is None:
        raise ValueError("Input image is empty")

    secret_id = os.getenv("TENCENT_SECRET_ID")
    secret_key = os.getenv("TENCENT_SECRET_KEY")
    
    if not secret_id or not secret_key:
        raise ValueError(
            "Missing credentials: set TENCENT_SECRET_ID and TENCENT_SECRET_KEY"
        )

    session_hash = req.session_hash if req is not None else "default"
    output_root = Path("./tmp/gradio") / str(session_hash) / "hunyuan_image3d"
    output_root.mkdir(parents=True, exist_ok=True)

    image_path = output_root / f"input_{seed}.png"
    image.save(image_path)
    if raw_image_cache is not None:
        raw_image_cache.save(output_root / "raw_image.png")

    glb_job_id = hunyuan_image3d_submit(
        image_path=str(image_path),
        secret_id=secret_id,
        secret_key=secret_key,
        result_format="GLB",
    )
    # glb_job_id = '1444588221266214912'
    # print("glb_job_id:", glb_job_id)
    glb_path, preview_path = query_and_download_hunyuan_job(
        job_id=glb_job_id,
        save_dir=str(output_root / "glb"),
        secret_id=secret_id,
        secret_key=secret_key,
    )

    return (
        glb_path,
        None,
        glb_path,
    )


def convert_glb_to_obj(glb_path: str, obj_dir: str) -> str:
    obj_dir_path = Path(obj_dir)
    obj_dir_path.mkdir(parents=True, exist_ok=True)
    obj_path = obj_dir_path / "result.obj"

    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    mesh.export(obj_path)

    return str(obj_path.resolve())

with gr.Blocks(delete_cache=(43200, 43200), theme=custom_theme) as demo:
    gr.HTML(image_css, visible=False)
    gr.HTML(lighting_css, visible=False)
    gr.Markdown(
        """
        ## ***EmbodiedGen***: Image-to-3D Asset
        **🔖 Version**: {VERSION}
        <p style="display: flex; gap: 10px; flex-wrap: nowrap;">
            <a href="https://horizonrobotics.github.io/EmbodiedGen">
                <img alt="📖 Documentation" src="https://img.shields.io/badge/📖-Documentation-blue">
            </a>
            <a href="https://arxiv.org/abs/2506.10600">
                <img alt="📄 arXiv" src="https://img.shields.io/badge/📄-arXiv-b31b1b">
            </a>
            <a href="https://github.com/HorizonRobotics/EmbodiedGen">
                <img alt="💻 GitHub" src="https://img.shields.io/badge/GitHub-000000?logo=github">
            </a>
            <a href="https://www.youtube.com/watch?v=rG4odybuJRk">
                <img alt="🎥 Video" src="https://img.shields.io/badge/🎥-Video-red">
            </a>
        </p>

        🖼️ Generate physically plausible 3D asset from single input image.
        """.format(
            VERSION=VERSION
        ),
        elem_classes=["header"],
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs() as input_tabs:
                with gr.Tab(
                    label="Image(auto seg)", id=0
                ) as single_image_input_tab:
                    raw_image_cache = gr.Image(
                        format="png",
                        image_mode="RGB",
                        type="pil",
                        visible=False,
                    )
                    image_prompt = gr.Image(
                        label="Input Image",
                        format="png",
                        image_mode="RGBA",
                        type="pil",
                        height=400,
                        elem_classes=["image_fit"],
                    )
                    gr.Markdown(
                        """
                        If you are not satisfied with the auto segmentation
                        result, please switch to the `Image(SAM seg)` tab."""
                    )
                with gr.Tab(
                    label="Image(SAM seg)", id=1
                ) as samimage_input_tab:
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_prompt_sam = gr.Image(
                                label="Input Image",
                                type="numpy",
                                height=400,
                                elem_classes=["image_fit"],
                            )
                            image_seg_sam = gr.Image(
                                label="SAM Seg Image",
                                image_mode="RGBA",
                                type="pil",
                                height=400,
                                visible=False,
                            )
                        with gr.Column(scale=1):
                            image_mask_sam = gr.AnnotatedImage(
                                elem_classes=["image_fit"]
                            )

                    fg_bg_radio = gr.Radio(
                        ["foreground_point", "background_point"],
                        label="Select foreground(green) or background(red) points, by default foreground",  # noqa
                        value="foreground_point",
                    )
                    gr.Markdown(
                        """ Click the `Input Image` to select SAM points,
                        after get the satisified segmentation, click `Generate`
                         button to generate the 3D asset. \n
                        Note: If the segmented foreground is too small relative
                         to the entire image area, the generation will fail.
                    """
                    )

            with gr.Accordion(label="Generation Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(
                        0, MAX_SEED, label="Seed", value=0, step=1
                    )
                    texture_size = gr.Slider(
                        1024,
                        4096,
                        label="UV texture size",
                        value=2048,
                        step=256,
                    )
                    rmbg_tag = gr.Radio(
                        choices=["rembg", "rmbg14"],
                        value=bg_rm_model_name,
                        label="Background Removal Model",
                    )
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed", value=False
                    )
                    project_delight = gr.Checkbox(
                        label="Back-project Delight",
                        value=True,
                    )
                gr.Markdown("Geo Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(
                        0.0,
                        10.0,
                        label="Guidance Strength",
                        value=7.5,
                        step=0.1,
                    )
                    ss_sampling_steps = gr.Slider(
                        1,
                        50,
                        label="Sampling Steps",
                        value=sample_step,
                        step=1,
                    )
                gr.Markdown("Visual Appearance Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(
                        0.0,
                        10.0,
                        label="Guidance Strength",
                        value=3.0,
                        step=0.1,
                    )
                    slat_sampling_steps = gr.Slider(
                        1,
                        50,
                        label="Sampling Steps",
                        value=sample_step,
                        step=1,
                    )

            generate_btn = gr.Button(
                "🚀 1. Generate(~2 mins)",
                variant="primary",
                interactive=False,
            )
            model_output_obj = gr.Textbox(label="raw mesh .obj", visible=False)
            # with gr.Row():
            #     extract_rep3d_btn = gr.Button(
            #         "🔍 2. Extract 3D Representation(~2 mins)",
            #         variant="primary",
            #         interactive=False,
            #     )
            with gr.Accordion(
                label="Enter Asset Attributes(optional)", open=False
            ):
                asset_cat_text = gr.Textbox(
                    label="Enter Asset Category (e.g., chair)"
                )
                height_range_text = gr.Textbox(
                    label="Enter **Height Range** in meter (e.g., 0.5-0.6)"
                )
                mass_range_text = gr.Textbox(
                    label="Enter **Mass Range** in kg (e.g., 1.1-1.2)"
                )
                asset_version_text = gr.Textbox(
                    label=f"Enter version (e.g., {VERSION})"
                )
            with gr.Row():
                extract_urdf_btn = gr.Button(
                    "🧩 2. Extract URDF with physics(~1 mins)",
                    variant="primary",
                    interactive=False,
                ) 
            with gr.Row():
                gr.Markdown(
                    "#### Estimated Asset 3D Attributes(No input required)"
                )
            with gr.Row():
                est_type_text = gr.Textbox(
                    label="Asset category", interactive=False
                )
                est_height_text = gr.Textbox(
                    label="Real height(.m)", interactive=False
                )
                est_mass_text = gr.Textbox(
                    label="Mass(.kg)", interactive=False
                )
                est_mu_text = gr.Textbox(
                    label="Friction coefficient", interactive=False
                )
            with gr.Row():
                download_urdf = gr.DownloadButton(
                    label="⬇️ 3. Download URDF",
                    variant="primary",
                    interactive=False,
                )

            gr.Markdown(
                """ NOTE: If `Asset Attributes` are provided, it will guide
                GPT to perform physical attributes restoration. \n
                The `Download URDF` file is restored to the real scale and
                has quality inspection, open with an editor to view details.
            """
            )
            enable_pre_resize = gr.State(enable_pre_resize)
            with gr.Row() as single_image_example:
                examples = gr.Examples(
                    label="Image Gallery",
                    examples=[
                        [image_path]
                        for image_path in sorted(
                            glob("apps/assets/example_image/*")
                        )
                    ],
                    inputs=[image_prompt, rmbg_tag, enable_pre_resize],
                    fn=preprocess_image_fn,
                    outputs=[image_prompt, raw_image_cache],
                    run_on_click=True,
                    examples_per_page=10,
                )

            with gr.Row(visible=False) as single_sam_image_example:
                examples = gr.Examples(
                    label="Image Gallery",
                    examples=[
                        [image_path]
                        for image_path in sorted(
                            glob("apps/assets/example_image/*")
                        )
                    ],
                    inputs=[image_prompt_sam],
                    fn=preprocess_sam_image_fn,
                    outputs=[image_prompt_sam, raw_image_cache],
                    run_on_click=True,
                    examples_per_page=10,
                )
        with gr.Column(scale=2):
            gr.Markdown("<br>")
            generated_model_output = gr.Model3D(
                label="Generated 3D Asset",
                height=700,
                interactive=False,
            )

    is_samimage = gr.State(False)
    output_buf = gr.State()
    selected_points = gr.State(value=[])

    demo.load(start_session)
    demo.unload(end_session)

    single_image_input_tab.select(
        lambda: tuple(
            [False, gr.Row.update(visible=True), gr.Row.update(visible=False)]
        ),
        outputs=[is_samimage, single_image_example, single_sam_image_example],
    )
    samimage_input_tab.select(
        lambda: tuple(
            [True, gr.Row.update(visible=True), gr.Row.update(visible=False)]
        ),
        outputs=[is_samimage, single_sam_image_example, single_image_example],
    )

    image_prompt.upload(
        preprocess_image_fn,
        inputs=[image_prompt, rmbg_tag, enable_pre_resize],
        outputs=[image_prompt, raw_image_cache],
    )
    image_prompt.change(
        lambda: tuple(
            [
                # gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                None,
                None,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        outputs=[
            # extract_rep3d_btn,
            extract_urdf_btn,
            download_urdf,
            generated_model_output,
            model_output_obj,
            asset_cat_text,
            height_range_text,
            mass_range_text,
            asset_version_text,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
        ],
    )
    image_prompt.change(
        active_btn_by_content,
        inputs=image_prompt,
        outputs=generate_btn,
    )

    image_prompt_sam.upload(
        preprocess_sam_image_fn,
        inputs=[image_prompt_sam],
        outputs=[image_prompt_sam, raw_image_cache],
    )
    image_prompt_sam.change(
        lambda: tuple(
            [
                # gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                None,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                None,
                [],
            ]
        ),
        outputs=[
            # extract_rep3d_btn,
            extract_urdf_btn,
            download_urdf,
            generated_model_output,
            model_output_obj,
            asset_cat_text,
            height_range_text,
            mass_range_text,
            asset_version_text,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
            image_mask_sam,
            selected_points,
        ],
    )

    image_prompt_sam.select(
        select_point,
        [
            image_prompt_sam,
            selected_points,
            fg_bg_radio,
        ],
        [image_mask_sam, image_seg_sam],
    )
    image_seg_sam.change(
        active_btn_by_content,
        inputs=image_seg_sam,
        outputs=generate_btn,
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).success(
        image_to_3d_via_hunyuan_api,
        inputs=[
            image_prompt,
            raw_image_cache,
            seed,
        ],
        outputs=[
            generated_model_output,
            model_output_obj,
            output_buf,
        ],
    ).success(
        lambda: gr.Button(interactive=True),
        outputs=[extract_urdf_btn],
    )

    extract_urdf_btn.click(
        extract_urdf_lazy,
        inputs=[
            output_buf,
            model_output_obj,
            asset_cat_text,
            height_range_text,
            mass_range_text,
            asset_version_text,
        ],
        outputs=[
            download_urdf,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
        ],
        queue=True,
        show_progress="full",
    ).success(
        lambda: gr.Button(interactive=True),
        outputs=[download_urdf],
    )


if __name__ == "__main__":
    demo.launch(server_port=8542)
