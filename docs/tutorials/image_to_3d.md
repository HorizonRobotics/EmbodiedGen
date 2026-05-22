# 🖼️ Image-to-3D: Physically Plausible 3D Asset Generation

Generate **physically plausible 3D assets** from a single input image, supporting **digital twin** and **simulation environments**.

---

## ⚡ Command-Line Usage
Three 3D generation backends are supported:

- [`SAM3D`](https://github.com/facebookresearch/sam-3d-objects) — local model (default)
- [`TRELLIS`](https://github.com/microsoft/TRELLIS) — local model
- `HUNYUAN3D` — Tencent Hunyuan3D Pro cloud API (no local GPU model needed)

Select the backend via `--image3d_model` (case-insensitive). Omit to use the default `SAM3D`.

```bash
img3d-cli --image_path apps/assets/example_image/sample_00.jpg \
apps/assets/example_image/sample_01.jpg \
--n_retry 2 --output_root outputs/imageto3d
```

### Using the Hunyuan3D Cloud Backend

Hunyuan3D Pro runs entirely on Tencent Cloud — useful when you don't have a local GPU. It requires Tencent Cloud Hunyuan3D `SecretId` / `SecretKey` and network access to `ai3d.tencentcloudapi.com` and the COS download host.

```bash
export TENCENT_SECRET_ID='your-secret-id'
export TENCENT_SECRET_KEY='your-secret-key'
img3d-cli --image3d_model HUNYUAN3D \
  --image_path apps/assets/example_image/sample_00.jpg \
  --n_retry 1 --output_root outputs/imageto3d_hunyuan
```

You will get the following results:

<div class="swiper swiper1" style="max-width: 1000px; margin: 20px auto; border-radius: 12px;">
  <div class="swiper-wrapper">
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/image2/sample_00.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/image2/sample_01.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/image2/sample_19.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
  </div>
  <div class="swiper-button-prev"></div>
  <div class="swiper-button-next"></div>
</div>


The generated results are organized as follows:
```sh
outputs/imageto3d/sample_xx/result
├── mesh
│   ├── material_0.png
│   ├── material.mtl
│   ├── sample_xx_collision.ply
│   ├── sample_xx.glb
│   ├── sample_xx_gs.ply
│   └── sample_xx.obj
├── sample_xx.urdf
└── video.mp4
```

- `mesh/` → Geometry and texture files, including visual mesh, collision mesh and 3DGS.
- `*.urdf` → Simulator-ready URDF with collision and visual meshes
- `video.mp4` → Preview of the generated 3D asset


!!! tip "Getting Started"
    - Try it directly online via our [Hugging Face Space](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Image-to-3D) — no installation required.
    - Explore EmbodiedGen generated sim-ready [Assets Gallery](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer).
    - For instructions on using the generated asset in any simulator, see [Any Simulators Tutorial](any_simulators.md).