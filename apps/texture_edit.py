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

os.environ["GRADIO_APP"] = "texture_edit"
import gradio as gr
from common import (
    MAX_SEED,
    VERSION,
    backproject_texture_v2,
    custom_theme,
    end_session,
    generate_condition,
    generate_texture_mvimages,
    get_seed,
    get_selected_image,
    image_css,
    lighting_css,
    render_result_video,
    start_session,
)


def active_btn_by_content(mesh_content: gr.Model3D, text_content: gr.Textbox):
    if (
        mesh_content is not None
        and text_content is not None
        and len(text_content) > 0
    ):
        interactive = True
    else:
        interactive = False

    return gr.Button(interactive=interactive)


with gr.Blocks(delete_cache=(43200, 43200), theme=custom_theme) as demo:
    gr.Markdown(
        """
        ## ***EmbodiedGen***: Texture Generation
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

        🎨 Generate visually rich textures for 3D mesh.

        """.format(
            VERSION=VERSION
        ),
        elem_classes=["header"],
    )
    gr.HTML(image_css)
    gr.HTML(lighting_css)
    with gr.Row():
        with gr.Column(scale=1):
            mesh_input = gr.Model3D(
                label="Upload Mesh File(.obj or .glb)", height=300
            )
            local_mesh = gr.Textbox(visible=False)
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
                "Note: The `reference image` is optional. If provided, please "
                "increase the `Condition Scale` in Generation Settings."
            )

            with gr.Accordion(label="Generation Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(
                        0, MAX_SEED, label="Seed", value=0, step=1
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed", value=False
                    )
                ip_adapt_scale = gr.Slider(
                    0, 1, label="IP-adapter Scale", value=0.7, step=0.05
                )
                cond_scale = gr.Slider(
                    0.0,
                    1.0,
                    label="Geo Condition Scale",
                    value=0.60,
                    step=0.01,
                )
                guidance_scale = gr.Slider(
                    1, 30, label="Text Guidance Scale", value=9, step=0.2
                )
                guidance_strength = gr.Slider(
                    0.0,
                    1.0,
                    label="Strength",
                    value=0.9,
                    step=0.05,
                )
                num_inference_steps = gr.Slider(
                    10, 100, label="Sampling Steps", value=50, step=5
                )
                texture_size = gr.Slider(
                    1024, 4096, label="UV texture size", value=2048, step=256
                )
                video_size = gr.Slider(
                    512, 2048, label="Video Resolution", value=512, step=256
                )

            generate_mv_btn = gr.Button(
                "🎨 1. Generate MV Images(~1min)",
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
                "Note: Select samples with consistent textures from various "
                "perspectives and no obvious reflections."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        dropdown = gr.Radio(
                            choices=["sample1", "sample2", "sample3"],
                            value="sample1",
                            label="Choose your favorite sample style.",
                        )
                        select_img = gr.Image(
                            visible=False,
                            format="png",
                            image_mode="RGBA",
                            type="filepath",
                            height=300,
                        )
                    with gr.Row():
                        project_delight = gr.Checkbox(
                            label="Back-project delight", value=True
                        )
                        fix_mesh = gr.Checkbox(
                            label="simplify mesh", value=False
                        )

                with gr.Column(scale=1):
                    texture_bake_btn = gr.Button(
                        "🛠️ 2. Texture Baking(~2min)",
                        variant="primary",
                        interactive=False,
                    )
                    download_btn = gr.DownloadButton(
                        label="⬇️ 3. Download Mesh",
                        variant="primary",
                        interactive=False,
                    )

            with gr.Row():
                mesh_output = gr.Model3D(
                    label="Mesh Edit Result",
                    clear_color=[0.8, 0.8, 0.8, 1],
                    height=380,
                    interactive=False,
                    elem_id="lighter_mesh",
                )
                mesh_outpath = gr.Textbox(visible=False)
                video_output = gr.Video(
                    label="Mesh Edit Video",
                    autoplay=True,
                    loop=True,
                    height=380,
                )

    with gr.Row():
        prompt_examples = []
        with open("apps/assets/example_texture/text_prompts.txt", "r") as f:
            for line in f:
                parts = line.strip().split("\\")
                prompt_examples.append([parts[0].strip(), parts[1].strip()])

        examples = gr.Examples(
            label="Mesh Gallery",
            examples=prompt_examples,
            inputs=[mesh_input, text_prompt],
            examples_per_page=10,
        )

    demo.load(start_session)
    demo.unload(end_session)

    mesh_input.change(
        lambda: tuple(
            [
                None,
                None,
                None,
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                None,
                None,
                None,
            ]
        ),
        outputs=[
            mesh_outpath,
            mesh_output,
            video_output,
            texture_bake_btn,
            download_btn,
            image_sample1,
            image_sample2,
            image_sample3,
        ],
    ).success(
        active_btn_by_content,
        inputs=[mesh_input, text_prompt],
        outputs=[generate_mv_btn],
    )

    text_prompt.change(
        active_btn_by_content,
        inputs=[mesh_input, text_prompt],
        outputs=[generate_mv_btn],
    )

    generate_mv_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).success(
        lambda: tuple(
            [
                None,
                None,
                None,
                gr.Button(interactive=False),
                gr.Button(interactive=False),
            ]
        ),
        outputs=[
            mesh_outpath,
            mesh_output,
            video_output,
            texture_bake_btn,
            download_btn,
        ],
    ).success(
        generate_condition,
        inputs=[mesh_input],
        outputs=[image_sample1, image_sample2, image_sample3],
    ).success(
        generate_texture_mvimages,
        inputs=[
            text_prompt,
            cond_scale,
            guidance_scale,
            guidance_strength,
            num_inference_steps,
            seed,
            ip_adapt_scale,
            ip_image,
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
        outputs=[texture_bake_btn],
    )

    texture_bake_btn.click(
        lambda: tuple([None, None, None, gr.Button(interactive=False)]),
        outputs=[mesh_outpath, mesh_output, video_output, download_btn],
    ).success(
        get_selected_image,
        inputs=[dropdown, usample1, usample2, usample3],
        outputs=select_img,
    ).success(
        backproject_texture_v2,
        inputs=[
            mesh_input,
            select_img,
            texture_size,
            project_delight,
            fix_mesh,
        ],
        outputs=[mesh_output, mesh_outpath, download_btn],
    ).success(
        lambda: gr.DownloadButton(interactive=True),
        outputs=[download_btn],
    ).success(
        render_result_video,
        inputs=[mesh_outpath, video_size],
        outputs=[video_output],
    )


if __name__ == "__main__":
    demo.launch()
