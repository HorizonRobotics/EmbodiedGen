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

import gradio as gr
import trimesh
from app_style import custom_theme, image_css, lighting_css
from embodied_gen.utils.tags import VERSION
from hunyuan_image3d_save import query_and_download_hunyuan_job
from hunyuan_text3d_submit import hunyuan_text3d_submit

# GRADIO_APP == "textto3d_sam3d", sam3d object model, by default.
# GRADIO_APP == "textto3d", TRELLIS model.
os.environ["GRADIO_APP"] = "textto3d_sam3d"

# Keep Gradio temp/cache under project workspace to avoid /tmp permission issues.
_gradio_tmp_root = Path(__file__).resolve().parent.parent / "tmp" / "gradio"
_gradio_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_gradio_tmp_root))
os.environ.setdefault("TMPDIR", str(_gradio_tmp_root))
tempfile.tempdir = str(_gradio_tmp_root)

MAX_SEED = 100000
SESSION_ROOT = Path("./tmp/gradio_sessions")
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

app_name = os.getenv("GRADIO_APP")
if app_name == "textto3d_sam3d":
    sample_step = 25
else:
    sample_step = 12


def start_session(req: gr.Request) -> None:
    session_hash = req.session_hash if req is not None else "default"
    (SESSION_ROOT / str(session_hash)).mkdir(parents=True, exist_ok=True)


def end_session(req: gr.Request) -> None:
    return None


def active_btn_by_text_content(content: str) -> gr.Button:
    return gr.Button(interactive=bool((content or "").strip()))


def get_seed(randomize_seed: bool, seed: int, max_seed: int = MAX_SEED) -> int:
    return random.randint(0, max_seed) if randomize_seed else seed


def text_to_3d_via_hunyuan_api(prompt: str, seed: int, req: gr.Request = None):
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Text prompt is empty")

    secret_id = os.getenv("TENCENT_SECRET_ID")
    secret_key = os.getenv("TENCENT_SECRET_KEY")
    
    if not secret_id or not secret_key:
        raise ValueError(
            "Missing credentials: set TENCENT_SECRET_ID and TENCENT_SECRET_KEY"
        )

    session_hash = req.session_hash if req is not None else "default"
    output_root = Path("./tmp/gradio") / str(session_hash) / "hunyuan_text3d"
    output_root.mkdir(parents=True, exist_ok=True)

    job_id = hunyuan_text3d_submit(
        prompt=prompt,
        secret_id=secret_id,
        secret_key=secret_key,
        result_format="GLB",
    )
    model_path, _ = query_and_download_hunyuan_job(
        job_id=job_id,
        save_dir=str(output_root / f"glb_{seed}"),
        secret_id=secret_id,
        secret_key=secret_key,
    )
    return model_path, model_path


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


def extract_urdf_lazy(
    glb_path: str,
    mesh_obj_path: str,
    asset_cat_text: str,
    height_range_text: str,
    mass_range_text: str,
    asset_version_text: str,
    req: gr.Request = None,
):
    from common import extract_urdf as _extract_urdf

    if (not mesh_obj_path) and glb_path:
        session_hash = req.session_hash if req is not None else "default"
        obj_dir = Path("./tmp/gradio") / str(session_hash) / "hunyuan_text3d" / "obj"
        mesh_obj_path = convert_glb_to_obj(glb_path, str(obj_dir))

    return _extract_urdf(
        glb_path,
        mesh_obj_path,
        asset_cat_text,
        height_range_text,
        mass_range_text,
        asset_version_text,
        req,
    )


with gr.Blocks(delete_cache=(43200, 43200), theme=custom_theme) as demo:
    gr.HTML(image_css, visible=False)
    gr.HTML(lighting_css, visible=False)
    gr.Markdown(
        """
        ## ***EmbodiedGen***: Text-to-3D Asset
        **🔖 Version**: {VERSION}

        📝 Create 3D assets from text descriptions.
        """.format(
            VERSION=VERSION
        ),
        elem_classes=["header"],
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_prompt = gr.Textbox(
                label="Text Prompt (Chinese or English)",
                placeholder="Input text prompt here",
            )

            with gr.Accordion(label="Generation Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    texture_size = gr.Slider(
                        1024, 4096, label="UV texture size", value=2048, step=256
                    )
                with gr.Row():
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                    project_delight = gr.Checkbox(
                        label="Back-project Delight", value=True
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
                        1, 50, label="Sampling Steps", value=sample_step, step=1
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
                        1, 50, label="Sampling Steps", value=sample_step, step=1
                    )

            generate_btn = gr.Button(
                "🚀 1. Generate 3D(~2 mins)",
                variant="primary",
                interactive=False,
            )
            model_output_obj = gr.Textbox(label="raw mesh .obj", visible=False)

            with gr.Accordion(label="Enter Asset Attributes(optional)", open=False):
                asset_cat_text = gr.Textbox(
                    label="Enter Asset Category (e.g., chair)"
                )
                height_range_text = gr.Textbox(
                    label="Enter Height Range in meter (e.g., 0.5-0.6)"
                )
                mass_range_text = gr.Textbox(
                    label="Enter Mass Range in kg (e.g., 1.1-1.2)"
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

            gr.Markdown("Estimated Asset 3D Attributes(No input required)")
            with gr.Row():
                est_type_text = gr.Textbox(
                    label="Asset category", interactive=False
                )
                est_height_text = gr.Textbox(
                    label="Real height(.m)", interactive=False
                )
                est_mass_text = gr.Textbox(label="Mass(.kg)", interactive=False)
                est_mu_text = gr.Textbox(
                    label="Friction coefficient", interactive=False
                )

            with gr.Row():
                download_urdf = gr.DownloadButton(
                    label="⬇️ 3. Download URDF",
                    variant="primary",
                    interactive=False,
                )

            prompt_examples = [
                "satin gold tea cup with saucer",
                "small bronze figurine of a lion",
                "brown leather bag",
                "Miniature cup with floral design",
                "带木质底座, 具有经纬线的地球仪",
                "橙色电动手钻, 有磨损细节",
                "手工制作的皮革笔记本",
            ]
            examples = gr.Examples(
                label="Gallery",
                examples=prompt_examples,
                inputs=[text_prompt],
                examples_per_page=10,
            )

        with gr.Column(scale=2):
            generated_model_output = gr.Model3D(
                label="Generated 3D Asset",
                height=700,
                interactive=False,
            )

    model_output_glb = gr.State("")

    demo.load(start_session)
    demo.unload(end_session)

    text_prompt.change(
        active_btn_by_text_content,
        inputs=[text_prompt],
        outputs=[generate_btn],
    )

    text_prompt.change(
        lambda: tuple(
            [
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
                "",
            ]
        ),
        outputs=[
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

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).success(
        text_to_3d_via_hunyuan_api,
        inputs=[text_prompt, seed],
        outputs=[generated_model_output, model_output_glb],
    ).success(
        lambda: gr.Button(interactive=True),
        outputs=[extract_urdf_btn],
    )

    extract_urdf_btn.click(
        extract_urdf_lazy,
        inputs=[
            model_output_glb,
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
    demo.launch(server_port=8082)
