# 🤖 Robot Learning

EmbodiedGen-generated worlds are not just viewable — they are **online training environments**. Spin up parallel `gym` environments from a generated layout, record sensor and trajectory data, and evaluate the grasp quality of generated assets.

In a companion [sim-to-real RL study](https://arxiv.org/abs/2603.18532), policies trained purely in EmbodiedGen-generated worlds reached **79.8%** simulation task success (from 9.7%) and **75.0%** real-robot task success (from 21.7%). The training code is not part of this repository; the tools below provide the environment side.

---

## 🏎️ Parallel Simulation Environments

Generate multiple parallel simulation environments with `gym.make` and record sensor and trajectory data.

```sh
python embodied_gen/scripts/parallel_sim.py \
--layout_file "outputs/layouts_gen/task_0000/layout.json" \
--output_dir "outputs/parallel_sim/task_0000" \
--num_envs 16
```

<div style="display: flex; justify-content: center; align-items: center; gap: 16px; margin: 16px 0;">
  <img src="../assets/parallel_sim.gif" alt="parallel_sim1"
       style="width: 330px; max-width: 100%; border-radius: 12px; display: block;">
  <img src="../assets/parallel_sim2.gif" alt="parallel_sim2"
       style="width: 330px; max-width: 100%; border-radius: 12px; display: block;">
</div>

The generated results are organized as follows:

```sh
outputs/parallel_sim/task_0000
├── 0.mp4               # Tiled preview render across the parallel envs
├── layout.json         # Copy of the layout the envs were built from
└── scene_tree.jpg      # Scene-graph visualization
```

---

## 🦾 Grasp-Quality Evaluation

Evaluate the grasp quality of a generated URDF via a top-down Panda grasp trial (ManiSkill + SAPIEN). Outputs a JSON report and per-trial MP4 next to the URDF.

```sh
python embodied_gen/scripts/eval_collision_success.py \
  --urdf-path outputs/imageto3d/sample_00/result/sample_00.urdf \
  --num-trials 4
```

---

!!! tip "Next Steps"
    - Compose task-driven worlds to train in — see [Layout Generation](layout_gen.md).
    - Export environments to your simulator — see [Any Simulators](any_simulators.md).
