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
  });
</script>

# Tutorials & Interface Usage

Welcome to the tutorials for `EmbodiedGen`. `EmbodiedGen` is a powerful toolset for generating 3D assets, textures, scenes, and interactive layouts ready for simulators and digital twin environments.

---

## ⚙️ Prerequisites

!!! tip "Prerequisites"
    Make sure to finish the [Installation Guide](../install.md) before starting tutorial. Missing dependencies will cause initialization errors. Model weights are automatically downloaded on first run.

---

## [🖼️ Image-to-3D](image_to_3d.md)

Generate **physically plausible 3D assets** from a single input image, supporting digital twin and simulation environments.


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

---

## [📝 Text-to-3D](text_to_3d.md)

Create **physically plausible 3D assets** from **text descriptions**, supporting a wide range of geometry, style, and material details.


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
    <div class="swiper-button-prev"></div>
    <div class="swiper-button-next"></div>
</div>

---

## [🎨 Texture Generation](texture_gen.md)

Generate **high-quality textures** for 3D meshes using **text prompts**, supporting both Chinese and English, to enhance the visual appearance of existing 3D assets.

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

## [🌍 3D Scene Generation](scene_gen.md)

Generate **physically consistent and visually coherent 3D environments** from text prompts. Typically used as **background** 3DGS scenes in simulators for efficient and photo-realistic rendering.

<img src="/assets/scene3d.gif" style="width: 500px; max-width: 100%; border-radius: 12px; display: block; margin: 16px auto;">

---

## [🏞️ Layout Generation](layout_gen.md)

Generate diverse, physically realistic, and scalable **interactive 3D scenes** from natural language task descriptions, while also modeling the robot and manipulable objects.

<div align="center" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; justify-items: center; margin: 20px 0;">
  <img src="/assets/layout1.gif" alt="layout1" style="width: 400px; border-radius: 12px; display: block;">
  <img src="/assets/layout2.gif" alt="layout2" style="width: 400px; border-radius: 12px; display: block;">
  <img src="/assets/layout3.gif" alt="layout3" style="width: 400px; border-radius: 12px; display: block;">
  <img src="/assets/Iscene_demo2.gif" alt="layout4" style="width: 400px; border-radius: 12px; display: block;">
</div>


---

## [🏎️ Parallel Simulation](gym_env.md)

Generate multiple **parallel simulation environments** with `gym.make` and record sensor and trajectory data.

<div style="display: flex; justify-content: center; align-items: center; gap: 16px; margin: 16px 0;">
  <img src="/assets/parallel_sim.gif" alt="parallel_sim1"
       style="width: 330px; max-width: 100%; border-radius: 12px; display: block;">
  <img src="/assets/parallel_sim2.gif" alt="parallel_sim2"
       style="width: 330px; max-width: 100%; border-radius: 12px; display: block;">
</div>


---

## [🎮 Use in Any Simulator](any_simulators.md)

Seamlessly use EmbodiedGen-generated assets in major simulators like **IsaacSim**, **MuJoCo**, **Genesis**, **PyBullet**, **IsaacGym**, and **SAPIEN**, featuring **accurate physical collisions** and **consistent visual appearance**.

<div align="center">
  <img src="/assets/simulators_collision.jpg" alt="simulators_collision" style="width: 400px; max-width: 100%; border-radius: 12px; display: block; margin: 16px 0;">
</div>

## [🔧 Real-to-Sim Digital Twin Creation](digital_twin.md)

<div align="center">
  <img src="/assets/real2sim_mujoco.gif" alt="real2sim_mujoco" style="width: 400px; max-width: 100%; border-radius: 12px; display: block; margin: 16px 0;">
</div>
