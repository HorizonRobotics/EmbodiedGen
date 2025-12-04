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

os.environ["GRADIO_APP"] = "imageto3d"
from glob import glob

import gradio as gr
from app_style import custom_theme, image_css, lighting_css
from common import (
    MAX_SEED,
    VERSION,
    active_btn_by_content,
    end_session,
    extract_3d_representations_v3,
    extract_urdf,
    get_seed,
    image_to_3d,
    preprocess_image_fn,
    preprocess_sam_image_fn,
    select_point,
    start_session,
)

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
        with gr.Column(scale=2):
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
                        value="rembg",
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

            with gr.Row() as single_image_example:
                examples = gr.Examples(
                    label="Image Gallery",
                    examples=[
                        [image_path]
                        for image_path in sorted(
                            glob("apps/assets/example_image/*")
                        )
                    ],
                    inputs=[image_prompt, rmbg_tag],
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
        with gr.Column(scale=1):
            gr.Markdown("<br>")
            video_output = gr.Video(
                label="Generated 3D Asset",
                autoplay=True,
                loop=True,
                height=300,
            )
            model_output_gs = gr.Model3D(
                label="Gaussian Representation", height=300, interactive=False
            )
            aligned_gs = gr.Textbox(visible=False)
            gr.Markdown(
                """ The rendering of `Gaussian Representation` takes additional 10s. """  # noqa
            )
            with gr.Row():
                model_output_mesh = gr.Model3D(
                    label="Mesh Representation",
                    height=300,
                    interactive=False,
                    clear_color=[0.8, 0.8, 0.8, 1],
                    elem_id="lighter_mesh",
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
        inputs=[image_prompt, rmbg_tag],
        outputs=[image_prompt, raw_image_cache],
    )
    image_prompt.change(
        lambda: tuple(
            [
                # gr.Button(interactive=False),
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
            ]
        ),
        outputs=[
            # extract_rep3d_btn,
            extract_urdf_btn,
            download_urdf,
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
                [],
            ]
        ),
        outputs=[
            # extract_rep3d_btn,
            extract_urdf_btn,
            download_urdf,
            model_output_gs,
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
        image_to_3d,
        inputs=[
            image_prompt,
            seed,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            raw_image_cache,
            image_seg_sam,
            is_samimage,
        ],
        outputs=[output_buf, video_output],
    ).success(
        extract_3d_representations_v3,
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
    demo.launch(server_port=8081)
