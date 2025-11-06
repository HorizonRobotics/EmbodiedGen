# 🏞️ Layout Generation — Interactive 3D Scenes

Layout Generation enables the generation of diverse, physically realistic, and scalable **interactive 3D scenes** directly from natural language task descriptions, while also modeling the robot's pose and relationships with manipulable objects. Target objects are randomly placed within the robot's reachable range, making the scenes readily usable for downstream simulation and reinforcement learning tasks in any mainstream simulator.

<div align="center" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; justify-items: center; margin: 20px 0;">
  <img src="../assets/layout1.gif" alt="layout1" style="width: 400px; border-radius: 12px; display: block;">
  <img src="../assets/layout2.gif" alt="layout2" style="width: 400px; border-radius: 12px; display: block;">
  <img src="../assets/layout3.gif" alt="layout3" style="width: 400px; border-radius: 12px; display: block;">
  <img src="../assets/Iscene_demo2.gif" alt="layout4" style="width: 400px; border-radius: 12px; display: block;">
</div>

!!! note "Model Requirement"
    The text-to-image model is based on `SD3.5 Medium`. Usage requires agreement to the [model license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).

---

## Prerequisites — Prepare Background 3D Scenes

Before running `layout-cli`, you need to prepare background 3D scenes.
You can either **generate your own** using the [`scene3d-cli`](scene_gen.md), or **download pre-generated backgrounds** for convenience.

Each scene takes approximately **30 minutes** to generate. For efficiency, we recommend pre-generating and listing them in `outputs/bg_scenes/scene_list.txt`.

```bash
# Option 1: Download pre-generated backgrounds (~4 GB)
hf download xinjjj/scene3d-bg --repo-type dataset --local-dir outputs

# Option 2: Download a larger background set (~14 GB)
hf download xinjjj..RLv2-BG --repo-type dataset --local-dir outputs
```

## Generate Interactive Layout Scenes

Use the `layout-cli` to create interactive 3D scenes based on task descriptions. Each layout generation takes approximately 30 minutes.

```sh
layout-cli \
  --task_descs "Place the pen in the mug on the desk" \
               "Put the fruit on the table on the plate" \
  --bg_list "outputs/bg_scenes/scene_list.txt" \
  --output_root "outputs/layouts_gen" \
  --insert_robot
```

You will get the following results:
<div align="center" style="display: flex; justify-content: center; align-items: flex-start; gap: 24px; margin: 20px auto; flex-wrap: wrap;">
  <img src="../assets/Iscene_demo1.gif" alt="Iscene_demo1"
       style="height: 200px; border-radius: 12px; display: block; width: auto;">
  <img src="../assets/Iscene_demo2.gif" alt="Iscene_demo2"
       style="height: 200px; border-radius: 12px; display: block; width: auto;">
</div>



### Batch Generation

You can also run multiple tasks via a task list file in the backend.

```sh
CUDA_VISIBLE_DEVICES=0 nohup layout-cli \
  --task_descs "apps/assets/example_layout/task_list.txt" \
  --bg_list "outputs/bg_scenes/scene_list.txt" \
  --output_root "outputs/layouts_gens" \
  --insert_robot > layouts_gens.log &
```

> 💡 Remove `--insert_robot` if you don’t need robot pose consideration in layout generation.

### Layout Randomization

Using `compose_layout.py`, you can **recompose the layout** of the generated interactive 3D scenes.

```sh
python embodied_gen/scripts/compose_layout.py \
--layout_path "outputs/layouts_gens/task_0000/layout.json" \
--output_dir "outputs/layouts_gens/task_0000/recompose" \
--insert_robot
```

### Load Interactive 3D Scenes in Simulators

We provide `sim-cli`, that allows users to easily load generated layouts into an interactive 3D simulation using the SAPIEN engine.

```sh
sim-cli --layout_path "outputs/layouts_gen/task_0000/layout.json" \
--output_dir "outputs/layouts_gen/task_0000/sapien_render" --insert_robot
```

!!! tip "Recommended Workflow"
    1. Generate or download background scenes using `scene3d-cli`.
    2. Create interactive layouts from task descriptions using `layout-cli`.
    3. Optionally recompose them using `compose_layout.py`.
    4. Load the final layouts into simulators with `sim-cli`.