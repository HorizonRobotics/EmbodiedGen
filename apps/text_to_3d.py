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

os.environ["GRADIO_APP"] = "textto3d"


import gradio as gr
from common import (
    MAX_SEED,
    VERSION,
    active_btn_by_text_content,
    custom_theme,
    end_session,
    extract_3d_representations_v2,
    extract_urdf,
    get_cached_image,
    get_seed,
    get_selected_image,
    image_css,
    image_to_3d,
    lighting_css,
    start_session,
    text2image_fn,
)

with gr.Blocks(delete_cache=(43200, 43200), theme=custom_theme) as demo:
    gr.Markdown(
        """
        ## ***EmbodiedGen***: Text-to-3D Asset
        **🔖 Version**: {VERSION}
        <p style="display: flex; gap: 10px; flex-wrap: nowrap;">
            <a href="https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html">
                <img alt="🌐 Project Page" src="https://img.shields.io/badge/🌐-Project_Page-blue">
            </a>
            <a href="https://arxiv.org/abs/xxxx.xxxxx">
                <img alt="📄 arXiv" src="https://img.shields.io/badge/📄-arXiv-b31b1b">
            </a>
            <a href="https://github.com/HorizonRobotics/EmbodiedGen">
                <img alt="💻 GitHub" src="https://img.shields.io/badge/GitHub-000000?logo=github">
            </a>
            <a href="https://www.youtube.com/watch?v=SnHhzHeb_aI">
                <img alt="🎥 Video" src="https://img.shields.io/badge/🎥-Video-red">
            </a>
        </p>

        📝 Create 3D assets from text descriptions for a wide range of geometry and styles.

        """.format(
            VERSION=VERSION
        ),
        elem_classes=["header"],
    )
    gr.HTML(image_css)
    gr.HTML(lighting_css)
    with gr.Row():
        with gr.Column(scale=1):
            raw_image_cache = gr.Image(
                format="png",
                image_mode="RGB",
                type="pil",
                visible=False,
            )
            text_prompt = gr.Textbox(
                label="Text Prompt (Chinese or English)",
                placeholder="Input text prompt here",
            )
            ip_image = gr.Image(
                label="Reference Image(optional)",
                format="png",
                image_mode="RGB",
                type="filepath",
                height=250,
                elem_classes=["image_fit"],
            )
            gr.Markdown(
                "Note: The `reference image` is optional, if use, "
                "please provide image in nearly square resolution."
            )

            with gr.Accordion(label="Image Generation Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(
                        0, MAX_SEED, label="Seed", value=0, step=1
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed", value=False
                    )
                rmbg_tag = gr.Radio(
                    choices=["rembg", "rmbg14"],
                    value="rembg",
                    label="Background Removal Model",
                )
                ip_adapt_scale = gr.Slider(
                    0, 1, label="IP-adapter Scale", value=0.3, step=0.05
                )
                img_guidance_scale = gr.Slider(
                    1, 30, label="Text Guidance Scale", value=12, step=0.2
                )
                img_inference_steps = gr.Slider(
                    10, 100, label="Sampling Steps", value=50, step=5
                )
                img_resolution = gr.Slider(
                    512,
                    1536,
                    label="Image Resolution",
                    value=1024,
                    step=128,
                )

            generate_img_btn = gr.Button(
                "🎨 1. Generate Images(~1min)",
                variant="primary",
                interactive=False,
            )
            dropdown = gr.Radio(
                choices=["sample1", "sample2", "sample3"],
                value="sample1",
                label="Choose your favorite sample style.",
            )
            select_img = gr.Image(
                visible=False,
                format="png",
                image_mode="RGBA",
                type="pil",
                height=300,
            )

            # text to 3d
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                texture_size = gr.Slider(
                    1024, 4096, label="UV texture size", value=2048, step=256
                )
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed", value=False
                    )
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
                        1, 50, label="Sampling Steps", value=12, step=1
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
                        1, 50, label="Sampling Steps", value=12, step=1
                    )

            generate_btn = gr.Button(
                "🚀 2. Generate 3D(~0.5 mins)",
                variant="primary",
                interactive=False,
            )
            model_output_obj = gr.Textbox(label="raw mesh .obj", visible=False)
            with gr.Row():
                extract_rep3d_btn = gr.Button(
                    "🔍 3. Extract 3D Representation(~1 mins)",
                    variant="primary",
                    interactive=False,
                )
            with gr.Accordion(
                label="Enter Asset Attributes(optional)", open=False
            ):
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
                    "🧩 4. Extract URDF with physics(~1 mins)",
                    variant="primary",
                    interactive=False,
                )
            with gr.Row():
                download_urdf = gr.DownloadButton(
                    label="⬇️ 5. Download URDF",
                    variant="primary",
                    interactive=False,
                )

        with gr.Column(scale=3):
            with gr.Row():
                image_sample1 = gr.Image(
                    label="sample1",
                    format="png",
                    image_mode="RGBA",
                    type="filepath",
                    height=300,
                    interactive=False,
                    elem_classes=["image_fit"],
                )
                image_sample2 = gr.Image(
                    label="sample2",
                    format="png",
                    image_mode="RGBA",
                    type="filepath",
                    height=300,
                    interactive=False,
                    elem_classes=["image_fit"],
                )
                image_sample3 = gr.Image(
                    label="sample3",
                    format="png",
                    image_mode="RGBA",
                    type="filepath",
                    height=300,
                    interactive=False,
                    elem_classes=["image_fit"],
                )
                usample1 = gr.Image(
                    format="png",
                    image_mode="RGBA",
                    type="filepath",
                    visible=False,
                )
                usample2 = gr.Image(
                    format="png",
                    image_mode="RGBA",
                    type="filepath",
                    visible=False,
                )
                usample3 = gr.Image(
                    format="png",
                    image_mode="RGBA",
                    type="filepath",
                    visible=False,
                )
            gr.Markdown(
                "The generated image may be of poor quality due to auto "
                "segmentation. Try adjusting the text prompt or seed."
            )
            with gr.Row():
                video_output = gr.Video(
                    label="Generated 3D Asset",
                    autoplay=True,
                    loop=True,
                    height=300,
                    interactive=False,
                )
                model_output_gs = gr.Model3D(
                    label="Gaussian Representation",
                    height=300,
                    interactive=False,
                )
                aligned_gs = gr.Textbox(visible=False)

                model_output_mesh = gr.Model3D(
                    label="Mesh Representation",
                    clear_color=[0.8, 0.8, 0.8, 1],
                    height=300,
                    interactive=False,
                    elem_id="lighter_mesh",
                )

            gr.Markdown("Estimated Asset 3D Attributes(No input required)")
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

    output_buf = gr.State()

    demo.load(start_session)
    demo.unload(end_session)

    text_prompt.change(
        active_btn_by_text_content,
        inputs=[text_prompt],
        outputs=[generate_img_btn],
    )

    generate_img_btn.click(
        lambda: tuple(
            [
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                None,
                "",
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
                None,
                None,
                None,
            ]
        ),
        outputs=[
            extract_rep3d_btn,
            extract_urdf_btn,
            download_urdf,
            generate_btn,
            model_output_gs,
            aligned_gs,
            model_output_mesh,
            video_output,
            asset_cat_text,
            height_range_text,
            mass_range_text,
            asset_version_text,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
            image_sample1,
            image_sample2,
            image_sample3,
        ],
    ).success(
        text2image_fn,
        inputs=[
            text_prompt,
            img_guidance_scale,
            img_inference_steps,
            ip_image,
            ip_adapt_scale,
            img_resolution,
            rmbg_tag,
            seed,
        ],
        outputs=[
            image_sample1,
            image_sample2,
            image_sample3,
            usample1,
            usample2,
            usample3,
        ],
    ).success(
        lambda: gr.Button(interactive=True),
        outputs=[generate_btn],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).success(
        get_selected_image,
        inputs=[dropdown, usample1, usample2, usample3],
        outputs=select_img,
    ).success(
        get_cached_image,
        inputs=[select_img],
        outputs=[raw_image_cache],
    ).success(
        image_to_3d,
        inputs=[
            select_img,
            seed,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            raw_image_cache,
        ],
        outputs=[output_buf, video_output],
    ).success(
        lambda: gr.Button(interactive=True),
        outputs=[extract_rep3d_btn],
    )

    extract_rep3d_btn.click(
        extract_3d_representations_v2,
        inputs=[
            output_buf,
            project_delight,
            texture_size,
        ],
        outputs=[
            model_output_mesh,
            model_output_gs,
            model_output_obj,
            aligned_gs,
        ],
    ).success(
        lambda: gr.Button(interactive=True),
        outputs=[extract_urdf_btn],
    )

    extract_urdf_btn.click(
        extract_urdf,
        inputs=[
            aligned_gs,
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
    demo.launch()
