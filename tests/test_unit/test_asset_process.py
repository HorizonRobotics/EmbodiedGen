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

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from plyfile import PlyData, PlyElement

asset_process = importlib.import_module(
    "embodied_gen.skills.asset-process.asset_process"
)


def _create_asset(
    root: Path,
    mesh_format: str = "obj",
    include_gaussian: bool = False,
) -> Path:
    asset_dir = root / "sample"
    result_dir = asset_dir / "result"
    mesh_dir = result_dir / "mesh"
    mesh_dir.mkdir(parents=True)

    mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    if mesh_format == "obj":
        mesh.export(mesh_dir / "sample.obj")
    elif mesh_format == "glb":
        mesh.export(mesh_dir / "sample.glb")
    elif mesh_format == "collision":
        mesh.export(mesh_dir / "sample_collision.obj")
    else:
        raise ValueError(f"Unsupported test mesh format: {mesh_format}")

    if include_gaussian:
        _write_gaussian_ply(mesh_dir / "sample_gs.ply")

    urdf_path = result_dir / "sample.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="sample">
  <link name="sample">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><mesh filename="mesh/sample.obj"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><mesh filename="mesh/sample_collision.obj"/></geometry>
    </collision>
    <extra_info>
      <min_height>0.15</min_height>
      <max_height>0.20</max_height>
      <real_height>0.175</real_height>
    </extra_info>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _write_gaussian_ply(path: Path) -> None:
    vertex = np.zeros(
        1,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ],
    )
    vertex["y"] = 1.0
    vertex["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def _create_flat_asset(root: Path) -> Path:
    asset_dir = root / "sample"
    mesh_dir = asset_dir / "mesh"
    mesh_dir.mkdir(parents=True)
    trimesh.creation.box(extents=(2.0, 4.0, 6.0)).export(
        mesh_dir / "sample.obj"
    )

    urdf_path = asset_dir / "sample.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="sample">
  <link name="sample">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="mesh/sample.obj"/></geometry>
    </visual>
    <extra_info>
      <min_height>0.15</min_height>
      <max_height>0.20</max_height>
      <real_height>0.175</real_height>
    </extra_info>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _read_heights(urdf_path: Path) -> dict[str, float]:
    root = ET.parse(urdf_path).getroot()
    extra_info = root.find("link/extra_info")
    assert extra_info is not None
    return {
        field: float(extra_info.findtext(field))
        for field in asset_process.URDF_HEIGHT_FIELDS
    }


def _read_height_text(urdf_path: Path) -> dict[str, str]:
    root = ET.parse(urdf_path).getroot()
    extra_info = root.find("link/extra_info")
    assert extra_info is not None
    return {
        field: extra_info.findtext(field)
        for field in asset_process.URDF_HEIGHT_FIELDS
    }


def test_process_asset_scales_rotates_and_recalculates_height(
    tmp_path: Path,
) -> None:
    source_urdf = _create_asset(tmp_path / "source")
    output_root = tmp_path / "output"

    output_urdf = asset_process.process_asset(
        urdf_path=source_urdf,
        scale_factor=2.0,
        rot_xyz=(90.0, 0.0, 0.0),
        output_dir=output_root,
    )

    assert output_urdf == output_root / "result/sample.urdf"
    output_mesh = trimesh.load(output_root / "result/mesh/sample.obj")
    np.testing.assert_allclose(output_mesh.extents, (4.0, 12.0, 8.0))
    assert _read_heights(output_urdf) == {
        "min_height": 12.0,
        "max_height": 12.0,
        "real_height": 12.0,
    }
    assert _read_height_text(output_urdf) == {
        "min_height": "12.0000",
        "max_height": "12.0000",
        "real_height": "12.0000",
    }

    root = ET.parse(output_urdf).getroot()
    assert root.find("link/visual/origin").get("rpy") == "1.5708 0 0"
    source_mesh = trimesh.load(source_urdf.parent / "mesh/sample.obj")
    np.testing.assert_allclose(source_mesh.extents, (2.0, 4.0, 6.0))


@pytest.mark.parametrize("mesh_format", ["glb", "collision"])
def test_height_calculation_falls_back_to_available_mesh(
    tmp_path: Path, mesh_format: str
) -> None:
    urdf_path = _create_asset(tmp_path / mesh_format, mesh_format)

    output_urdf = asset_process.process_asset(
        urdf_path=urdf_path,
        inplace=True,
    )

    assert output_urdf == urdf_path
    assert _read_heights(output_urdf) == {
        "min_height": 4.0,
        "max_height": 4.0,
        "real_height": 4.0,
    }


def test_height_calculation_requires_a_mesh(tmp_path: Path) -> None:
    urdf_path = _create_asset(tmp_path / "missing")
    (urdf_path.parent / "mesh/sample.obj").unlink()

    with pytest.raises(FileNotFoundError, match="height calculation"):
        asset_process.process_asset(urdf_path=urdf_path, inplace=True)


def test_process_asset_supports_scale_only(tmp_path: Path) -> None:
    urdf_path = _create_asset(tmp_path / "scale_only")

    output_urdf = asset_process.process_asset(
        urdf_path=urdf_path,
        scale_factor=0.5,
        inplace=True,
    )

    assert output_urdf == urdf_path
    mesh = trimesh.load(urdf_path.parent / "mesh/sample.obj")
    np.testing.assert_allclose(mesh.extents, (1.0, 2.0, 3.0))
    assert _read_heights(output_urdf) == {
        "min_height": 2.0,
        "max_height": 2.0,
        "real_height": 2.0,
    }


def test_gaussian_processing_extends_legacy_scale_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urdf_path = _create_asset(
        tmp_path / "gaussian",
        include_gaussian=True,
    )
    calls: list[tuple[str, object]] = []

    class FakeGaussian:
        @classmethod
        def load_from_ply(cls, path: str) -> "FakeGaussian":
            calls.append(("load", path))
            return cls()

        @staticmethod
        def trans_to_quatpose(matrix: np.ndarray) -> np.ndarray:
            calls.append(("pose", matrix.copy()))
            return np.array([0.0, 0.0, 0.0, 1.0])

        def rescale(self, scale: float) -> None:
            calls.append(("rescale", scale))

        def get_gaussians(self, instance_pose: np.ndarray) -> "FakeGaussian":
            calls.append(("rotate", instance_pose.copy()))
            return self

        def save_to_ply(self, path: str) -> None:
            calls.append(("save", path))

    monkeypatch.setattr(
        asset_process,
        "import_module",
        lambda name: SimpleNamespace(GaussianOperator=FakeGaussian),
    )

    asset_process.process_asset(
        urdf_path=urdf_path,
        scale_factor=2.0,
        rot_xyz=(90.0, 0.0, 0.0),
        inplace=True,
    )

    assert [name for name, _ in calls] == [
        "load",
        "rescale",
        "pose",
        "rotate",
        "save",
    ]
    assert calls[1] == ("rescale", 2.0)
    np.testing.assert_allclose(
        calls[2][1],
        asset_process.Rotation.from_euler(
            "xyz", (90.0, 0.0, 0.0), degrees=True
        ).as_matrix(),
    )


def test_collision_loader_preserves_multiple_obj_groups(
    tmp_path: Path,
) -> None:
    collision_path = tmp_path / "multi_collision.obj"
    collision_path.write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
v 1 0 1
v 0 1 1
o lower
f 1 2 3
o upper
f 4 5 6
""",
        encoding="utf-8",
    )

    meshes = asset_process.AssetProcessor._load_collision_obj(
        str(collision_path)
    )

    assert len(meshes) == 2
    assert all(len(mesh.faces) == 1 for mesh in meshes)


def test_gaussian_file_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urdf_path = _create_asset(tmp_path / "without_gaussian")

    def fail_import(name: str) -> None:
        raise AssertionError(f"Unexpected GS module import: {name}")

    monkeypatch.setattr(asset_process, "import_module", fail_import)

    output_urdf = asset_process.process_asset(
        urdf_path=urdf_path,
        scale_factor=0.5,
        inplace=True,
    )

    assert output_urdf == urdf_path


def test_process_asset_supports_flat_asset_layout(tmp_path: Path) -> None:
    urdf_path = _create_flat_asset(tmp_path / "source")
    output_root = tmp_path / "processed"

    output_urdf = asset_process.process_asset(
        urdf_path=urdf_path,
        rot_xyz=(90.0, 0.0, 0.0),
        output_dir=output_root,
    )

    assert output_urdf == output_root / "sample.urdf"
    output_mesh = trimesh.load(output_root / "mesh/sample.obj")
    np.testing.assert_allclose(output_mesh.extents, (2.0, 6.0, 4.0))
    assert _read_heights(output_urdf) == {
        "min_height": 6.0,
        "max_height": 6.0,
        "real_height": 6.0,
    }


def test_keep_urdf_raw_rot_composes_inverse_rotation(tmp_path: Path) -> None:
    urdf_path = _create_asset(tmp_path / "source")
    output_dir = tmp_path / "processed"
    baked_rotation = asset_process.Rotation.from_euler(
        "xyz",
        (20.0, 35.0, 10.0),
        degrees=True,
    )
    original_rotation = asset_process.Rotation.from_euler(
        "xyz",
        (1.5708, 0.0, 0.0),
        degrees=False,
    )

    output_urdf = asset_process.process_asset(
        urdf_path=urdf_path,
        rot_xyz=(20.0, 35.0, 10.0),
        keep_urdf_raw_rot=True,
        output_dir=output_dir,
    )

    root = ET.parse(output_urdf).getroot()
    for path in ("link/visual/origin", "link/collision/origin"):
        origin = root.find(path)
        rpy_text = origin.get("rpy")
        assert all(
            len(component.rsplit(".", maxsplit=1)[-1]) == 4
            for component in rpy_text.split()
        )
        updated_rpy = np.fromstring(rpy_text, sep=" ")
        updated_rotation = asset_process.Rotation.from_euler(
            "xyz",
            updated_rpy,
            degrees=False,
        )
        np.testing.assert_allclose(
            (updated_rotation * baked_rotation).as_matrix(),
            original_rotation.as_matrix(),
            atol=1e-4,
        )


def test_output_dir_cannot_replace_source_without_inplace(
    tmp_path: Path,
) -> None:
    urdf_path = _create_flat_asset(tmp_path / "source")

    with pytest.raises(ValueError, match="use inplace=True"):
        asset_process.AssetProcessor(
            urdf_path=urdf_path,
            output_dir=urdf_path.parent,
        )
