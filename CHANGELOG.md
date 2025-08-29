# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0).

## [0.1.4] - 2025-08-29
### Feature
- Supports importing the output of the EmbodiedGen Layout Generation module with a single `gym.make` command to create multiple parallel simulation environments and interactive 3D worlds.
- Supports coacd-based mesh decomposition for better efficiency and accuracy in physical simulations.

## [0.1.3] - 2025-08-19
### Feature
- Release `layout-cli` 🏞️ Layout(Interactive 3D Worlds) Generation, generate interactive 3D scene from task description.


## [0.1.2] - 2025-07-31
### Feature
- Release `scene3d-cli` 🌍 3D Scene Generation, generate a color mesh and 3DGS scene from text description.
- Release docker file and docker image.


## [0.1.1] - 2025-07-11
### Feature
- Added intelligent quality checkers and auto-retry pipeline for `image-to-3d` and `text-to-3d`.
- Added unit tests for quality checkers.
- `text-to-3d` now supports more `text-to-image` models, pipeline success rate improved to 94%.

## [0.1.0] - 2025-07-04
### Feature
🖼️ Single Image to Physics Realistic 3D Asset
- Generates watertight, simulation-ready 3D meshes with physical attributes.
- Auto-labels semantic and quality tags (geometry, texture, foreground, etc.).
- Produces 2K textures with highlight removal and multi-view fusion.

📝 Text-to-3D Asset Generation
- Creates realistic 3D assets from natural language (English & Chinese).
- Filters assets via QA tags to ensure visual and geometric quality.

🎨 Texture Generation & Editing
- Generates 2K textures from mesh and text with semantic alignment.
- Plug-and-play modules adapt text-to-image models for 3D textures.
- Supports realistic and stylized texture outputs, including text textures.

