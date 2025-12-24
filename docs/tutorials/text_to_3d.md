# 📝 Text-to-3D: Generate 3D Assets from Text

Create **physically plausible 3D assets** from **text descriptions**, supporting a wide range of geometry, style, and material details.

---

## ⚡ Command-Line Usage

**Basic CLI(recommend)**

Text-to-image model based on Stable Diffusion 3.5 Medium， English prompts only. Usage requires agreement to the [model license (click “Accept”)](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).

```bash
text3d-cli \
  --prompts "small bronze figurine of a lion" "A globe with wooden base" "wooden table with embroidery" \
  --n_image_retry 1 \
  --n_asset_retry 1 \
  --n_pipe_retry 1 \
  --seed_img 0 \
  --output_root outputs/textto3d
```

- `--n_image_retry`: Number of retries per prompt for text-to-image generation
- `--n_asset_retry`: Retry attempts for image-to-3D assets generation
- `--n_pipe_retry`: Pipeline retry for end-to-end 3D asset quality check
- `--seed_img`: Optional initial seed image for style guidance
- `--output_root`: Directory to save generated assets

For large-scale 3D asset generation, set `--n_image_retry=4` `--n_asset_retry=3` `--n_pipe_retry=2`, slower but better, via automatic checking and retries. For more diverse results, omit `--seed_img`.

You will get the following results:

<div class="swiper swiper1" style="max-width: 1000px; margin: 20px auto; border-radius: 12px;">
    <div class="swiper-wrapper">
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text2/sample3d_0.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px; border-radius: 12px;"
            >
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"small bronze figurine of a lion"</p>
        </div>
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text2/sample3d_1.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px;">
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"A globe with wooden base"</p>
        </div>
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text2/sample3d_2.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px;">
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"wooden table with embroidery"</p>
        </div>
    </div>
    <div class="swiper-button-prev swiper1-prev"></div>
    <div class="swiper-button-next swiper1-next"></div>
</div>

---


Kolors Model CLI (Supports Chinese & English Prompts):
```bash
bash embodied_gen/scripts/textto3d.sh \
    --prompts "A globe with wooden base and latitude and longitude lines" "橙色电动手钻，有磨损细节" \
    --output_root outputs/textto3d_k
```

> Models with more permissive licenses can be found in `embodied_gen/models/image_comm_model.py`.


The generated results are organized as follows:
```sh
outputs/textto3d
├── asset3d
│   ├── sample3d_xx
│   │   └── result
│   │       ├── mesh
│   │       │   ├── material_0.png
│   │       │   ├── material.mtl
│   │       │   ├── sample3d_xx_collision.obj
│   │       │   ├── sample3d_xx.glb
│   │       │   ├── sample3d_xx_gs.ply
│   │       │   └── sample3d_xx.obj
│   │       ├── sample3d_xx.urdf
│   │       └── video.mp4
└── images
    ├── sample3d_xx.png
    ├── sample3d_xx_raw.png
```

- `mesh/` → 3D geometry and texture files for the asset, including visual mesh, collision mesh and 3DGS
- `*.urdf` → Simulator-ready URDF including collision and visual meshes
- `video.mp4` → Preview video of the generated 3D asset
- `images/sample3d_xx.png` → Foreground-extracted image used for image-to-3D step
- `images/sample3d_xx_raw.png` → Original generated image from the text-to-image step

---

!!! tip "Getting Started"
    - You can also try Text-to-3D instantly online via our [Hugging Face Space](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D) — no installation required.
    - Explore EmbodiedGen generated sim-ready [Assets Gallery](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer).
    - For instructions on using the generated asset in any simulator, see [Any Simulators Tutorial](any_simulators.md).