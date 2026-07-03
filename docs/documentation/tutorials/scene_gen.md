# 🌍 3D Scene Generation

Generate **physically consistent and visually coherent 3D environments** from text prompts. Typically used as **background** 3DGS scenes in simulators for efficient and photo-realistic rendering.

---

<img src="../assets/scene3d.gif" style="width: 600px; border-radius: 12px; display: block; margin: 16px auto;">

---

## ⚡ Command-Line Usage

> 💡 Run `bash install.sh scene3d` to install additional dependencies if you plan to use `scene3d-cli`.

It typically takes ~30 minutes per scene to generate both the colored mesh and 3D Gaussian Splat(3DGS) representation.

```bash
CUDA_VISIBLE_DEVICES=0 scene3d-cli \
  --prompts "Art studio with easel and canvas" \
  --output_dir outputs/bg_scenes/ \
  --seed 0 \
  --gs3d.max_steps 4000 \
  --disable_pano_check
```

The generated results are organized as follows:
```sh
outputs/bg_scenes/scene_000
├── gs_model.ply
├── gsplat_cfg.yml
├── mesh_model.ply
├── pano_image.png
├── prompt.txt
└── video.mp4
```

- `gs_model.ply` → Generated 3D scene in 3D Gaussian Splat representation.
- `mesh_model.ply` → Color mesh representation of the generated scene.
- `gsplat_cfg.yml` → Configuration file for 3DGS training and rendering parameters.
- `pano_image.png` → Generated panoramic view image.
- `prompt.txt` → Original scene generation prompt for traceability.
- `video.mp4` → Preview RGB and depth preview of the generated 3D scene.

!!! note "Usage Notes"
    - `3D Scene Generation` produces background 3DGS scenes optimized for efficient rendering in simulation environments. We also provide hybrid rendering examples combining background 3DGS with foreground interactive assets, see the [example]()
    for details.
    - In Layout Generation, we further demonstrate task-desc-driven interactive 3D scene generation, building complete 3D scenes based on natural language task descriptions. See the [Layout Generation Guide](layout_gen.md).
