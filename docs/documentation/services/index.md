# Interactive 3D Generation & Visualization Services

EmbodiedGen provides a suite of **interactive services** that transform images and text into **physically plausible, simulator-ready 3D assets**.
Each service is optimized for visual quality, simulation compatibility, and scalability — making it easy to create, edit, and explore assets for **digital twin**, **robotic simulation**, and **AI embodiment** scenarios.

---


!!! tip "Prerequisites"
    Make sure to finish the [Installation Guide](../install.md) before launching any service. Missing dependencies will cause initialization errors. Model weights are automatically downloaded on first run.

---

## 🧩 Overview of Available Services

| Service | Description |
|----------|--------------|
| [🖼️ **Image to 3D**](image_to_3d.md) | Generate physically plausible 3D asset URDF from single input image, offering high-quality support for digital twin systems. |
| [📝 **Text to 3D**](text_to_3d.md) | Generate physically plausible 3D assets from text descriptions for a wide range of geometry and styles. |
| [🎨 **Texture Edit**](texture_edit.md) | Generate visually rich textures for existing 3D meshes. |
| [📸 **Asset Gallery**](visualize_asset.md) | Explore and download EmbodiedGen All-Simulators-Ready Assets. |

---

## ⚙️ How to Run Locally

!!! tip "Quick Start"
    Each service can be launched directly as a local Gradio app:
    ```bash
    # Example: Run the Image-to-3D service
    python apps/image_to_3d.py
    ```

    Models are automatically downloaded on first run. For full CLI usage, please check the corresponding [tutorials](../tutorials/index.md).

---

## 🧭 Next Steps

- [📘 Tutorials](../tutorials/index.md) – Learn how to use EmbodiedGen in generating interactive 3D scenes for embodied intelligence.
- [🧱 API Reference](../api/index.md) – Integrate EmbodiedGen code programmatically.
