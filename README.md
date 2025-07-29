# *EmbodiedGen*: Towards a Generative 3D World Engine for Embodied Intelligence

[![🌐 Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html)
[![📄 arXiv](https://img.shields.io/badge/📄-arXiv-b31b1b)](https://arxiv.org/abs/2506.10600)
[![🎥 Video](https://img.shields.io/badge/🎥-Video-red)](https://www.youtube.com/watch?v=rG4odybuJRk)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Image_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Image-to-3D)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Text_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Texture_Gen_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen)
[![中文介绍](https://img.shields.io/badge/中文介绍-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/HH1cPBhK2xcDbyCK4BBTbw)


> ***EmbodiedGen*** is a generative engine to create diverse and interactive 3D worlds composed of high-quality 3D assets(mesh & 3DGS) with plausible physics, leveraging generative AI to address the challenges of generalization in embodied intelligence related research.
> It composed of six key modules: `Image-to-3D`, `Text-to-3D`, `Texture Generation`, `Articulated Object Generation`, `Scene Generation` and `Layout Generation`.

<img src="apps/assets/overall.jpg" alt="Overall Framework" width="700"/>

---

## ✨ Table of Contents of EmbodiedGen
- [🖼️ Image-to-3D](#image-to-3d)
- [📝 Text-to-3D](#text-to-3d)
- [🎨 Texture Generation](#texture-generation)
- [🌍 3D Scene Generation](#3d-scene-generation)
- [⚙️ Articulated Object Generation](#articulated-object-generation)
- [🏞️ Layout (Interactive 3D Worlds) Generation](#layout-generation)

## 🚀 Quick Start

### ✅ Setup Environment
```sh
git clone https://github.com/HorizonRobotics/EmbodiedGen.git
cd EmbodiedGen
git checkout v0.1.3-nightly
git submodule update --init --recursive --progress
conda create -n embodiedgen python=3.10.13 -y # recommended to use a new env.
conda activate embodiedgen
bash install.sh basic
```

### ✅ Setup GPT Agent

Update the API key in file: `embodied_gen/utils/gpt_config.yaml`.

You can choose between two backends for the GPT agent:

- **`gpt-4o`** (Recommended) – Use this if you have access to **Azure OpenAI**.
- **`qwen2.5-vl`** – An alternative with free usage via OpenRouter, apply a free key [here](https://openrouter.ai/settings/keys) and update `api_key` in `embodied_gen/utils/gpt_config.yaml` (50 free requests per day)


---

<h2 id="image-to-3d">🖼️ Image-to-3D</h2>

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Image_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Image-to-3D) Generate physically plausible 3D asset URDF from single input image, offering high-quality support for digital twin systems.

<img src="apps/assets/image_to_3d.jpg" alt="Image to 3D" width="900">

### ☁️ Service
Run the image-to-3D generation service locally.
Models downloaded automatically on first run, please be patient.
```sh
# Run in foreground
python apps/image_to_3d.py
# Or run in the background
CUDA_VISIBLE_DEVICES=0 nohup python apps/image_to_3d.py > /dev/null 2>&1 &
```

### ⚡ API
Generate physically plausible 3D assets from image input via the command-line API.
```sh
img3d-cli --image_path apps/assets/example_image/sample_04.jpg apps/assets/example_image/sample_19.jpg \
    --n_retry 2 --output_root outputs/imageto3d

# See result(.urdf/mesh.obj/mesh.glb/gs.ply) in ${output_root}/sample_xx/result
```

---


<h2 id="text-to-3d">📝 Text-to-3D</h2>

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Text_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D) Create 3D assets from text descriptions for a wide range of geometry and styles.

<img src="apps/assets/text_to_3d.jpg" alt="Text to 3D" width="900">

### ☁️ Service
Deploy the text-to-3D generation service locally.

Text-to-image model based on the Kolors model, supporting Chinese and English prompts.
Models downloaded automatically on first run, please be patient.
```sh
python apps/text_to_3d.py
```

### ⚡ API
Text-to-image model based on SD3.5 Medium, English prompts only.
Usage requires agreement to the [model license(click accept)](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), models downloaded automatically.

For large-scale 3D assets generation, set `--n_pipe_retry=2` to ensure high end-to-end 3D asset usability through automatic quality check and retries. For more diverse results, do not set `--seed_img`.

```sh
text3d-cli --prompts "small bronze figurine of a lion" "A globe with wooden base" "wooden table with embroidery" \
    --n_image_retry 2 --n_asset_retry 2 --n_pipe_retry 1 --seed_img 0 \
    --output_root outputs/textto3d
```

Text-to-image model based on the Kolors model.
```sh
bash embodied_gen/scripts/textto3d.sh \
    --prompts "small bronze figurine of a lion" "A globe with wooden base and latitude and longitude lines" "橙色电动手钻，有磨损细节" \
    --output_root outputs/textto3d_k
```
ps: models with more permissive licenses found in `embodied_gen/models/image_comm_model.py`

---


<h2 id="texture-generation">🎨 Texture Generation</h2>

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Texture_Gen_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen) Generate visually rich textures for 3D mesh.

<img src="apps/assets/texture_gen.jpg" alt="Texture Gen" width="900">


### ☁️ Service
Run the texture generation service locally.
Models downloaded automatically on first run, see `download_kolors_weights`, `geo_cond_mv`.
```sh
python apps/texture_edit.py
```

### ⚡ API
Support Chinese and English prompts.
```sh
bash embodied_gen/scripts/texture_gen.sh \
    --mesh_path "apps/assets/example_texture/meshes/robot_text.obj" \
    --prompt "举着牌子的写实风格机器人，大眼睛，牌子上写着“Hello”的文字" \
    --output_root "outputs/texture_gen/robot_text"

bash embodied_gen/scripts/texture_gen.sh \
    --mesh_path "apps/assets/example_texture/meshes/horse.obj" \
    --prompt "A gray horse head with flying mane and brown eyes" \
    --output_root "outputs/texture_gen/gray_horse"
```

---

<h2 id="3d-scene-generation">🌍 3D Scene Generation</h2>

<img src="apps/assets/scene3d.gif" alt="scene3d" style="width: 640px;">

### ⚡ API
> Run `bash install.sh extra` to install additional requirements if you need to use `scene3d-cli`.

It takes ~30mins to generate a color mesh and 3DGS per scene.

```sh
CUDA_VISIBLE_DEVICES=0 scene3d-cli \
--prompts "Art studio with easel and canvas" \
--output_dir outputs/bg_scenes/ \
--seed 0 \
--gs3d.max_steps 4000 \
--disable_pano_check
```

---


<h2 id="articulated-object-generation">⚙️ Articulated Object Generation</h2>

🚧 *Coming Soon*

<img src="apps/assets/articulate.gif" alt="articulate" style="width: 480px;">


---


<h2 id="layout-generation">🏞️ Layout(Interactive 3D Worlds) Generation</h2>

### 💬 Generate Layout from task description

<table>
  <tr>
    <td><img src="apps/assets/layout1.gif" alt="layout1" width="360"/></td>
    <td><img src="apps/assets/layout2.gif" alt="layout2" width="360"/></td>
  </tr>
  <tr>
    <td><img src="apps/assets/layout3.gif" alt="layout3" width="360"/></td>
    <td><img src="apps/assets/layout4.gif" alt="layout4" width="360"/></td>
  </tr>
</table>

Text-to-image model based on SD3.5 Medium, usage requires agreement to the [model license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). All models auto-downloaded at the first run.

You can generate any desired room as background using `scene3d-cli`. As each scene takes approximately 30 minutes to generate, we recommend pre-generating them for efficiency and adding them to `outputs/bg_scenes/scene_list.txt`.

We provided some sample background assets created with `scene3d-cli`. Download them(~4G) using `hf download xinjjj/scene3d-bg --repo-type dataset --local-dir outputs`.

Generating one interactive 3D scene from task description with `layout-cli` takes approximately 30 minutes.
```sh
CUDA_VISIBLE_DEVICES=0 nohup layout-cli \
--task_descs "Place the pen in the mug on the desk" "Move the lamp to the left" \
--bg_list "outputs/bg_scenes/scene_list.txt" --output_root "outputs/layouts_gen" > layouts_gen.log &
```

<table>
  <tr>
    <td><img src="apps/assets/Iscene_demo1.gif" alt="Iscene_demo1" width="300"/></td>
    <td><img src="apps/assets/Iscene_demo2.gif" alt="Iscene_demo2" width="300"/></td>
  </tr>
</table>

We provide `sim-cli`, that allows users to easily load generated layouts into an interactive 3D simulation using the SAPIEN engine (will support for more simulators in future updates).

```sh
sim-cli --layout_path "outputs/layouts_gen/task_0000/layout.json" \
--output_dir "outputs/layouts_gen/task_0000/sapien_render"
```

Using `compose_layout.py`, you can recompose the layout of the generated interactive 3D scenes. (Support for texture editing and augmentation will be added later.)
```sh
python embodied_gen/scripts/compose_layout.py \
--layout_path "outputs/layouts_gen/task_0000/layout.json" \
--output_dir "outputs/layouts_gen/task_0000_recompose"
```

### 🖼️ Real-to-Sim Digital Twin

<img src="apps/assets/real2sim_mujoco.gif" alt="real2sim_mujoco" style="width: 512px;">

---

## For Developer
```sh
pip install -e .[dev] && pre-commit install
python -m pytest # Pass all unit-test are required.
```

## 📚 Citation

If you use EmbodiedGen in your research or projects, please cite:

```bibtex
@misc{wang2025embodiedgengenerative3dworld,
      title={EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence},
      author={Xinjie Wang and Liu Liu and Yu Cao and Ruiqi Wu and Wenkang Qin and Dehui Wang and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2506.10600},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.10600},
}
```

---

## 🙌 Acknowledgement

EmbodiedGen builds upon the following amazing projects and models:
🌟 [Trellis](https://github.com/microsoft/TRELLIS) | 🌟 [Hunyuan-Delight](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0) | 🌟 [Segment Anything](https://github.com/facebookresearch/segment-anything) | 🌟 [Rembg](https://github.com/danielgatis/rembg) | 🌟 [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | 🌟 [Stable Diffusion x4](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) | 🌟 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 🌟 [Kolors](https://github.com/Kwai-Kolors/Kolors) | 🌟 [ChatGLM3](https://github.com/THUDM/ChatGLM3) | 🌟 [Aesthetic Score](http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html) | 🌟 [Pano2Room](https://github.com/TrickyGo/Pano2Room) | 🌟 [Diffusion360](https://github.com/ArcherFMY/SD-T2I-360PanoImage) | 🌟 [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) | 🌟 [diffusers](https://github.com/huggingface/diffusers) | 🌟 [gsplat](https://github.com/nerfstudio-project/gsplat) | 🌟 [QWEN-2.5VL](https://github.com/QwenLM/Qwen2.5-VL) | 🌟 [GPT4o](https://platform.openai.com/docs/models/gpt-4o) | 🌟 [SD3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

---

## ⚖️ License

This project is licensed under the [Apache License 2.0](LICENSE). See the `LICENSE` file for details.
