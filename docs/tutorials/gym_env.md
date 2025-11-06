# Simulation in Parallel Envs

Generate multiple parallel simulation environments with `gym.make` and record sensor and trajectory data.

---

## ⚡ Command-Line Usage

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

