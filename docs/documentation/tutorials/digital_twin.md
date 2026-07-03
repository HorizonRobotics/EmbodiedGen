# 🔧 Real-to-Sim Digital Twin Creation

Recreate real-world scenes in simulation: capture a photo of each real object, generate its **physically plausible 3D twin** with [Image-to-3D](image_to_3d.md), and compose the twins into a simulator scene ([Any Simulators](any_simulators.md)).

```sh
img3d-cli --image_path path/to/real_capture.jpg --output_root outputs/digital_twin
```

Each asset keeps metric scale, collision geometry, and physical properties inferred from the image, so the resulting scene behaves consistently with its real counterpart — below, an EmbodiedGen digital twin replayed in MuJoCo.

<img src="../assets/real2sim_mujoco.gif" alt="real2sim_mujoco" width="600">
