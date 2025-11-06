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


# 🎨 Texture Generation Service

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Texture_Gen_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen)

This service launches a web application to generate visually rich textures for 3D mesh.

<div class="swiper swiper1" style="max-width: 1000px; margin: 20px auto; border-radius: 12px;">
  <div class="swiper-wrapper">
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/hello2.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/love4.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/robot_china.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/horse1.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/horse2.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/shoe_0_0.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/shoe_0_3.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/clock_num.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/clock5.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/vase1.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/vase2.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/drill1.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
    <div class="swiper-slide model-card">
      <model-viewer
        src="https://raw.githubusercontent.com/HochCC/ShowCase/main/edit/drill4.glb"
        auto-rotate
        camera-controls
        style="display:block; width:100%; height:250px; background-color: #f8f8f8;">
      </model-viewer>
    </div>
  </div>
  <div class="swiper-button-prev"></div>
  <div class="swiper-button-next"></div>
</div>

---

## ☁️ Run the App Service

!!! note "Note"
    Gradio servive is a simplified demonstration. For the full functionality, please refer to [texture-cli](../tutorials/texture_edit.md).

Run the texture generation service locally. Models downloaded automatically on first run, see `download_kolors_weights`, `geo_cond_mv`.


```sh
# Run in foreground
python apps/texture_edit.py

# Or run in the background
CUDA_VISIBLE_DEVICES=0 nohup python apps/texture_edit.py > /dev/null 2>&1 &
```

---

!!! tip "Getting Started"
    - Try it directly online via our [Hugging Face Space](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Texture-Gen) — no installation required.
    - Explore EmbodiedGen generated sim-ready [Assets Gallery](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer).
    - For instructions on using the generated asset in any simulator, see [Any Simulators Tutorial](tutorials/any_simulators.md).
