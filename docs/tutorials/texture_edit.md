# 🎨 Texture Generation: Create Visually Rich Textures for 3D Meshes

Generate **high-quality textures** for 3D meshes using **text prompts**, supporting both **Chinese and English**. This allows you to enhance the visual appearance of existing 3D assets for simulation, visualization, or digital twin applications.

---

## ⚡ Command-Line Usage

```bash
texture-cli \
  --mesh_path "apps/assets/example_texture/meshes/robot_text.obj" \
              "apps/assets/example_texture/meshes/horse.obj" \
  --prompt "举着牌子的写实风格机器人，大眼睛，牌子上写着“Hello”的文字" \
           "A gray horse head with flying mane and brown eyes" \
  --output_root "outputs/texture_gen" \
  --seed 0
```

- `--mesh_path` → Path(s) to input 3D mesh files
- `--prompt` → Text prompt(s) describing desired texture/style for each mesh
- `--output_root` → Directory to save textured meshes and related outputs
- `--seed` → Random seed for reproducible texture generation


You will get the following results:

<div class="swiper swiper1" style="max-width: 1000px; margin: 20px auto; border-radius: 12px;">
  <div class="swiper-wrapper">
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit2/robot_text.glb"
        auto-rotate
        camera-controls
        camera-orbit="180deg auto auto"
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit2/horse.glb"
        auto-rotate
        camera-controls
        camera-orbit="90deg auto auto"
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
  </div>
  <div class="swiper-button-prev"></div>
  <div class="swiper-button-next"></div>
</div>

---

!!! tip "Getting Started"
    - Try it directly online via our [Hugging Face Space](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen) — no installation required.
    - Explore EmbodiedGen generated sim-ready [Assets Gallery](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer).
    - For instructions on using the generated asset in any simulator, see [Any Simulators Tutorial](any_simulators.md).