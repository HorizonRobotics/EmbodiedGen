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


import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import tyro
from scipy.spatial.transform import Rotation


@dataclass
class GenesisClothConvertConfig:
    input_path: str
    output_dir: str
    scale: float | None = None
    overwrite: bool = False


DEFAULT_CLOTH_MATERIAL: dict[str, float | bool] = {
    "rho": 0.2,
    "static_friction": 0.5,
    "kinetic_friction": 0.4,
    "stretch_compliance": 1e-9,
    "bending_compliance": 1e-6,
    "stretch_relaxation": 0.5,
    "bending_relaxation": 0.2,
    "air_resistance": 1e-3,
    "particle_size": 0.02,
    "self_collision": True,
}


def convert_cloth_to_genesis(
    input_path: str,
    output_dir: str,
    scale: float | None = None,
    overwrite: bool = False,
) -> str:
    """Convert one URDF visual mesh to a Genesis cloth asset package.

    Args:
        input_path: URDF path containing a visual mesh reference.
        output_dir: Directory where the ``genesis`` package is written.
        scale: Optional uniform scale recorded in the manifest and used by the
            simulator. The exported mesh itself is not rescaled.
        overwrite: Whether to replace an existing Genesis package.

    Returns:
        Path to the generated manifest JSON file.
    """

    urdf_path = Path(input_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"Input URDF does not exist: {urdf_path}")
    if urdf_path.suffix.lower() != ".urdf":
        raise ValueError(f"input_path must point to a URDF file: {urdf_path}")

    source_mesh, urdf_transform = _resolve_urdf_visual_mesh(urdf_path)
    package_root = Path(output_dir)
    genesis_dir = package_root / "genesis"
    if genesis_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Genesis cloth package already exists: {genesis_dir}. "
            "Pass overwrite=True to replace it."
        )
    genesis_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh(source_mesh)
    mesh, warnings = _clean_mesh(mesh)
    mesh.apply_transform(urdf_transform["matrix"])
    warnings.append("applied URDF visual origin transform to cloth mesh")

    cloth_mesh_path = genesis_dir / "cloth_mesh.obj"
    material_path = genesis_dir / "cloth_material.json"
    manifest_path = genesis_dir / "manifest.json"

    mesh.export(cloth_mesh_path)
    _write_json(material_path, DEFAULT_CLOTH_MATERIAL)

    mesh_stats = _mesh_stats(mesh)
    warnings.extend(_mesh_warnings(mesh_stats))
    manifest = {
        "asset_name": source_mesh.stem,
        "source_urdf": _relative_path(urdf_path, package_root),
        "source_mesh": _relative_path(source_mesh, package_root),
        "cloth_mesh": _relative_path(cloth_mesh_path, package_root),
        "material": _relative_path(material_path, package_root),
        "scale": 1.0 if scale is None else float(scale),
        "up_axis": "z",
        "urdf_visual_origin": urdf_transform["origin"],
        "mesh_stats": mesh_stats,
        "warnings": warnings,
    }
    _write_json(manifest_path, manifest)
    return str(manifest_path)


def entrypoint(**kwargs) -> str:
    if kwargs:
        cfg = GenesisClothConvertConfig(**kwargs)
    else:
        cfg = tyro.cli(GenesisClothConvertConfig)

    manifest_path = convert_cloth_to_genesis(
        input_path=cfg.input_path,
        output_dir=cfg.output_dir,
        scale=cfg.scale,
        overwrite=cfg.overwrite,
    )
    print(manifest_path)
    return manifest_path


def _resolve_urdf_visual_mesh(urdf_path: Path) -> tuple[Path, dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    for visual_tag in root.findall(".//visual"):
        mesh_tag = visual_tag.find("geometry/mesh")
        if mesh_tag is None:
            continue
        filename = mesh_tag.get("filename")
        if not filename:
            raise ValueError(
                f"Visual mesh in URDF has no filename: {urdf_path}"
            )

        source_mesh = Path(filename)
        if not source_mesh.is_absolute():
            source_mesh = urdf_path.parent / source_mesh
        if not source_mesh.exists():
            raise FileNotFoundError(
                f"Visual mesh does not exist: {source_mesh}"
            )

        origin_tag = visual_tag.find("origin")
        xyz, rpy = _parse_origin(origin_tag)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        matrix[:3, 3] = xyz
        return source_mesh, {
            "matrix": matrix,
            "origin": {
                "urdf": urdf_path.name,
                "xyz": [round(float(value), 6) for value in xyz],
                "rpy": [round(float(value), 6) for value in rpy],
            },
        }

    raise ValueError(f"No visual mesh found in URDF: {urdf_path}")


def _parse_origin(
    origin_tag: ET.Element | None,
) -> tuple[np.ndarray, np.ndarray]:
    if origin_tag is None:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    xyz = np.fromiter(
        (float(value) for value in origin_tag.get("xyz", "0 0 0").split()),
        dtype=float,
        count=3,
    )
    rpy = np.fromiter(
        (float(value) for value in origin_tag.get("rpy", "0 0 0").split()),
        dtype=float,
        count=3,
    )
    return xyz, rpy


def _load_mesh(source_mesh: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(source_mesh, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {source_mesh}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(
            f"Expected a Trimesh from {source_mesh}, got {type(loaded)}"
        )
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Source mesh is empty: {source_mesh}")
    return loaded


def _clean_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    mesh = mesh.copy()
    warnings: list[str] = []
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)

    if not mesh.is_watertight:
        warnings.append(
            "mesh is not watertight; this is usually acceptable for cloth"
        )

    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    else:
        mesh.remove_degenerate_faces()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()

    if (
        len(mesh.vertices) != original_vertices
        or len(mesh.faces) != original_faces
    ):
        warnings.append(
            "mesh cleanup changed topology from "
            f"{original_vertices} vertices/{original_faces} faces to "
            f"{len(mesh.vertices)} vertices/{len(mesh.faces)} faces"
        )
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh cleanup produced an empty cloth mesh")
    return mesh, warnings


def _mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    bbox_min, bbox_max = mesh.bounds
    extents = np.asarray(mesh.extents, dtype=float)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_min": bbox_min.astype(float).round(6).tolist(),
        "bbox_max": bbox_max.astype(float).round(6).tolist(),
        "extents": extents.round(6).tolist(),
        "watertight": bool(mesh.is_watertight),
    }


def _mesh_warnings(mesh_stats: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if mesh_stats["faces"] > 20000:
        warnings.append(
            "mesh has more than 20000 faces; Genesis cloth simulation may be slow"
        )
    extents = np.asarray(mesh_stats["extents"], dtype=float)
    if np.any(extents <= 1e-6):
        warnings.append("mesh has a near-zero bounding-box extent")
    if float(extents.max()) > 10.0:
        warnings.append(
            "mesh bounding box is larger than 10 meters; check scale"
        )
    return warnings


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    entrypoint()
