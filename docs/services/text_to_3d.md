# 📝 Text-to-3D Service

[![🤗 Hugging Face](https://img.shields.io/badge/🤗-Text_to_3D_Demo-blue)](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D)

This service launches a web application to generate physically plausible 3D assets from text descriptions for a wide range of geometry and styles.

<div class="swiper swiper1" style="max-width: 1000px; margin: 20px auto; border-radius: 12px;">
    <div class="swiper-wrapper">
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text/c2.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px; border-radius: 12px;"
            >
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"Antique brass key, intricate filigree"</p>
        </div>
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text/c3.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px;">
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"Rusty old wrench, peeling paint"</p>
        </div>
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text/c4.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px;">
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"Sleek black drone, red sensors"</p>
        </div>
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text/c7.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px;">
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"Miniature screwdriver with bright orange handle"</p>
        </div>
        <div class="swiper-slide model-card">
        <model-viewer
            src="https://raw.githubusercontent.com/HochCC/ShowCase/main/text/c9.glb"
            auto-rotate
            camera-controls
            background-color="#ffffff"
            style="display:block; width: 100%; height: 160px;">
        </model-viewer>
        <p style="text-align: center; margin-top: 8px; font-size: 14px;">"European style wooden dressing table"</p>
        </div>
    </div>
    <div class="swiper-button-prev swiper1-prev"></div>
    <div class="swiper-button-next swiper1-next"></div>
</div>

---

## ☁️ Run the App Service

Create 3D assets from text descriptions for a wide range of geometry and styles.
!!! note "Note"
    Gradio servive is a simplified demonstration. For the full functionality, please refer to [text3d-cli](../tutorials/text_to_3d.md).


Text-to-image model based on the Kolors model, supporting Chinese and English prompts. Models downloaded automatically on first run, please be patient.

```sh
# Run in foreground
python apps/text_to_3d.py

# Or run in the background
CUDA_VISIBLE_DEVICES=0 nohup python apps/text_to_3d.py > /dev/null 2>&1 &
```

---

!!! tip "Getting Started"
    - You can also try Text-to-3D instantly online via our [Hugging Face Space](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Text-to-3D) — no installation required.
    - Explore EmbodiedGen generated sim-ready [Assets Gallery](https://huggingface.co/spaces/HorizonRobotics/EmbodiedGen-Gallery-Explorer).
    - For instructions on using the generated asset in any simulator, see [Any Simulators Tutorial](tutorials/any_simulators.md).