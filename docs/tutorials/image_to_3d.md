<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/swiper/swiper-bundle.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper/swiper-bundle.min.css">
<script>
  document.addEventListener('DOMContentLoaded', function () {
    const swiper = new Swiper('.swiper1', {
      loop: true,
      slidesPerView: 3,
      spaceBetween: 20,
      navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
      },
      centeredSlides: false,
      noSwiping: true,
      noSwipingClass: 'swiper-no-swiping',
      watchSlidesProgress: true,
    });
    const modelViewers = document.querySelectorAll('model-viewer');
    let loadedCount = 0;
    modelViewers.forEach(mv => {
      mv.addEventListener('load', () => {
        loadedCount++;
        if (loadedCount === modelViewers.length) {
          swiper.update();
        }
      });
    });
  });
</script>

# 🖼️ Image-to-3D: Physically Plausible 3D Asset Generation

Generate **physically plausible 3D assets** from a single input image, supporting **digital twin** and **simulation environments**.

---

## ⚡ Command-Line Usage

```bash
img3d-cli --image_path apps/assets/example_image/sample_00.jpg \
apps/assets/example_image/sample_01.jpg apps/assets/example_image/sample_19.jpg \
--n_retry 1 --output_root outputs/imageto3d
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