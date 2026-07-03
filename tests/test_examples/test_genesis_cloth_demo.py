# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Smoke tests for the Genesis cloth demo helpers."""

import importlib.util
import json

import pytest
import trimesh
from embodied_gen.scripts.flexible_obj.genesis_cloth_converter import (
    convert_cloth_to_genesis,
)
from embodied_gen.scripts.flexible_obj.simulate_genesis_cloth import (
    simulate_genesis_cloth,
)


def _write_square_cloth(path):
    mesh = trimesh.Trimesh(
        vertices=[
            [-0.25, -0.25, 0.0],
            [0.25, -0.25, 0.0],
            [0.25, 0.25, 0.0],
            [-0.25, 0.25, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    mesh.export(path)


def _write_cloth_urdf(path):
    path.write_text(
        """<robot name=\"cloth\">
  <link name=\"cloth\">
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"1.5708 0 0\"/>
      <geometry>
        <mesh filename=\"mesh/sample.obj\"/>
      </geometry>
    </visual>
  </link>
</robot>
"""
    )


def test_convert_cloth_to_genesis(tmp_path):
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    _write_square_cloth(mesh_dir / "sample.obj")

    urdf_path = tmp_path / "sample.urdf"
    output_dir = tmp_path / "converted"
    _write_cloth_urdf(urdf_path)

    manifest_path = convert_cloth_to_genesis(
        str(urdf_path), output_dir=str(output_dir), overwrite=True
    )
    manifest = json.loads(
        (output_dir / "genesis" / "manifest.json").read_text()
    )

    assert manifest_path == str(output_dir / "genesis" / "manifest.json")
    assert (output_dir / manifest["cloth_mesh"]).exists()
    assert (output_dir / manifest["material"]).exists()
    assert manifest["source_urdf"].endswith("sample.urdf")
    assert manifest["urdf_visual_origin"]["rpy"] == [1.5708, 0.0, 0.0]
    assert manifest["mesh_stats"]["vertices"] == 4
    assert manifest["mesh_stats"]["faces"] == 2


@pytest.mark.skipif(
    importlib.util.find_spec("genesis") is None,
    reason="Genesis is not installed",
)
def test_simulate_genesis_cloth_smoke(tmp_path):
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    _write_square_cloth(mesh_dir / "sample.obj")
    urdf_path = tmp_path / "sample.urdf"
    _write_cloth_urdf(urdf_path)
    output_dir = tmp_path / "converted"
    convert_cloth_to_genesis(
        str(urdf_path), output_dir=str(output_dir), overwrite=True
    )

    video_path = simulate_genesis_cloth(
        str(output_dir),
        output_dir=str(tmp_path / "genesis_sim"),
        init_height=0.5,
        duration_seconds=0.4,
        sim_steps=2,
        shake=True,
        shake_steps=2,
        shake_amplitude=0.001,
        device="cpu",
        headless=True,
        camera_res=(80, 80),
        render_interval=1,
        fps=5,
    )

    assert (tmp_path / "genesis_sim" / "run_summary.json").exists()
    assert (tmp_path / "genesis_sim" / "sim_config.json").exists()
    assert video_path.endswith("video.mp4")
