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

"""Asset processing utilities for scaling and rotating 3D assets."""

import math
import shutil
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro
from scipy.spatial.transform import Rotation
from embodied_gen.utils.log import logger

__all__ = [
    "AssetProcessConfig",
    "AssetProcessor",
    "entrypoint",
    "process_asset",
]

URDF_HEIGHT_FIELDS = ("min_height", "max_height", "real_height")
URDF_RESULT_DIR = "result"
MESH_DIR = "mesh"
HEIGHT_AXIS = 1


@dataclass
class AssetProcessConfig:
    """Configuration for asset scaling and rotation.

    Args:
        urdf_path: Path to the URDF file to process.
        scale_factor: Positive uniform scaling factor.
        rot_xyz: XYZ Euler rotation in degrees.
        keep_urdf_raw_rot: Whether to preserve the original URDF rotation.
        inplace: Whether to modify the source asset directly.
        output_dir: Target asset directory used when ``inplace`` is false.
    """

    urdf_path: str
    scale_factor: float = 1.0
    rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    keep_urdf_raw_rot: bool = False
    inplace: bool = False
    output_dir: Optional[str] = None


class AssetProcessor:
    """Scale and rotate mesh, collision, and Gaussian-splat asset files."""

    def __init__(
        self,
        urdf_path: str | Path,
        scale_factor: float = 1.0,
        rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        keep_urdf_raw_rot: bool = False,
        output_dir: Optional[str | Path] = None,
        inplace: bool = False,
    ) -> None:
        """Initialize an asset processor.

        Args:
            urdf_path: Path to the URDF file to process.
            scale_factor: Positive uniform scaling factor.
            rot_xyz: XYZ Euler rotation in degrees.
            keep_urdf_raw_rot: Whether to preserve the original visual and
                collision rotation after baking the asset rotation.
            output_dir: Target asset directory used when ``inplace`` is false.
            inplace: Whether to modify the source asset directly.

        Raises:
            FileNotFoundError: If the URDF file does not exist.
            ValueError: If the transform or output configuration is invalid.
        """
        self.urdf_path = Path(urdf_path)
        self.scale_factor = float(scale_factor)
        self.rot_xyz = tuple(float(value) for value in rot_xyz)
        self.keep_urdf_raw_rot = keep_urdf_raw_rot
        self.inplace = inplace

        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        if not math.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError(
                f"Scale factor must be positive and finite, got: "
                f"{self.scale_factor}"
            )
        if len(self.rot_xyz) != 3 or not all(
            math.isfinite(value) for value in self.rot_xyz
        ):
            raise ValueError(
                "rot_xyz must contain three finite degree values, got: "
                f"{self.rot_xyz}"
            )

        if self.urdf_path.parent.name == URDF_RESULT_DIR:
            self.asset_dir = self.urdf_path.parent.parent
            self.result_dir = Path(URDF_RESULT_DIR)
        else:
            self.asset_dir = self.urdf_path.parent
            self.result_dir = Path()
        self.node_name = self.urdf_path.stem
        self.rotation = Rotation.from_euler("xyz", self.rot_xyz, degrees=True)
        self.rotation_transform = np.eye(4)
        self.rotation_transform[:3, :3] = self.rotation.as_matrix()

        if self.inplace:
            self.output_dir = self.asset_dir.parent
        else:
            if output_dir is None:
                raise ValueError("output_dir is required when inplace=False")
            self.output_dir = Path(output_dir)
            if self.output_dir.resolve() == self.asset_dir.resolve():
                raise ValueError(
                    "output_dir must differ from the source asset directory; "
                    "use inplace=True to modify the source asset"
                )

    def process(self) -> Path:
        """Run the complete asset processing workflow.

        Returns:
            Path to the processed URDF file.
        """
        if self.inplace:
            output_asset_dir = self.asset_dir
            output_urdf_path = self.urdf_path
        else:
            output_asset_dir = self.output_dir
            output_urdf_path = self._copy_asset_structure(output_asset_dir)

        self._process_asset_files_parallel(output_asset_dir)
        actual_height = self._calculate_actual_height(output_asset_dir)
        self._write_urdf_height(output_urdf_path, actual_height)
        if self.keep_urdf_raw_rot:
            self._update_urdf_origins(output_urdf_path)

        logger.info(
            "Processed %s with scale=%s, rot_xyz=%s -> %s",
            self.asset_dir,
            self.scale_factor,
            self.rot_xyz,
            output_asset_dir,
        )
        return output_urdf_path

    def _copy_asset_structure(self, output_asset_dir: Path) -> Path:
        """Copy the complete source asset to the output directory."""
        shutil.rmtree(output_asset_dir, ignore_errors=True)
        shutil.copytree(self.asset_dir, output_asset_dir)
        return output_asset_dir / self.result_dir / f"{self.node_name}.urdf"

    def _process_asset_files_parallel(self, output_asset_dir: Path) -> None:
        """Apply the configured transform to every supported asset format."""
        mesh_dir = output_asset_dir / self.result_dir / MESH_DIR

        # Keep the legacy asset-scale processing paths separate because each
        # format has different loading and export behavior.
        tasks = [
            (mesh_dir / f"{self.node_name}.obj", self._scale_obj_mesh),
            (mesh_dir / f"{self.node_name}.glb", self._scale_glb_mesh),
            (
                mesh_dir / f"{self.node_name}_collision.obj",
                self._scale_collision_mesh,
            ),
            (
                mesh_dir / f"{self.node_name}_gs.ply",
                self._scale_gaussian_splat,
            ),
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(task, path) for path, task in tasks]
            for future in futures:
                future.result()  # Propagate any exceptions

    def _scale_obj_mesh(self, mesh_path: Path) -> None:
        """Scale and rotate an OBJ mesh."""
        if not mesh_path.exists():
            return

        mesh = trimesh.load(str(mesh_path))
        mesh.apply_scale(self.scale_factor)
        self._apply_mesh_rotation(mesh)
        mesh.export(str(mesh_path))

    def _scale_glb_mesh(self, mesh_path: Path) -> None:
        """Scale and rotate every geometry in a GLB scene."""
        if not mesh_path.exists():
            return

        mesh = trimesh.load(str(mesh_path))
        for mesh_part in mesh.geometry.values():
            mesh_part.apply_scale(self.scale_factor)
            self._apply_mesh_rotation(mesh_part)
        mesh.export(str(mesh_path))

    def _scale_collision_mesh(self, mesh_path: Path) -> None:
        """Scale and rotate a potentially multi-object collision OBJ."""
        if not mesh_path.exists():
            return

        meshes = self._load_collision_obj(str(mesh_path))
        scene = trimesh.Scene()
        for mesh_part in meshes:
            mesh_part.apply_scale(self.scale_factor)
            self._apply_mesh_rotation(mesh_part)
            scene.add_geometry(mesh_part)
        scene.export(str(mesh_path))

    def _scale_gaussian_splat(self, mesh_path: Path) -> None:
        """Scale and rotate a Gaussian splatting model."""
        if not mesh_path.exists():
            return

        gaussian_operator = import_module(
            "embodied_gen.models.gs_model"
        ).GaussianOperator
        gs_model = gaussian_operator.load_from_ply(str(mesh_path))
        gs_model.rescale(self.scale_factor)
        if not self._is_identity_rotation:
            instance_pose = gaussian_operator.trans_to_quatpose(
                self.rotation.as_matrix()
            )
            gs_model = gs_model.get_gaussians(instance_pose=instance_pose)
        gs_model.save_to_ply(str(mesh_path))

    @property
    def _is_identity_rotation(self) -> bool:
        """Return whether the configured rotation is an identity operation."""
        return np.allclose(self.rotation.as_matrix(), np.eye(3))

    def _apply_mesh_rotation(self, mesh: trimesh.Trimesh) -> None:
        """Apply the configured rotation after the legacy scale operation."""
        if self._is_identity_rotation:
            return
        mesh.apply_transform(self.rotation_transform)

    def _calculate_actual_height(self, output_asset_dir: Path) -> float:
        """Calculate Y-axis height from the best available transformed mesh."""
        mesh_dir = output_asset_dir / self.result_dir / MESH_DIR
        candidates = (
            mesh_dir / f"{self.node_name}.obj",
            mesh_dir / f"{self.node_name}.glb",
            mesh_dir / f"{self.node_name}_collision.obj",
        )
        for mesh_path in candidates:
            if not mesh_path.exists():
                continue
            scene = trimesh.load(str(mesh_path), force="scene")
            geometry = scene.to_geometry()
            if len(geometry.vertices) == 0:
                continue
            height = float(np.ptp(geometry.vertices[:, HEIGHT_AXIS]))
            if not math.isfinite(height):
                raise ValueError(
                    f"Calculated non-finite height from mesh: {mesh_path}"
                )
            return height

        candidate_text = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "No mesh is available for height calculation. Expected one of: "
            f"{candidate_text}"
        )

    @staticmethod
    def _write_urdf_height(urdf_path: Path, height: float) -> None:
        """Write actual mesh height to all URDF height metadata fields."""
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        link = root.find("link")
        if link is None:
            raise ValueError(f"No link element found in URDF: {urdf_path}")

        extra_info = link.find("extra_info")
        if extra_info is None:
            extra_info = ET.SubElement(link, "extra_info")

        value = f"{height:.4f}"
        for field in URDF_HEIGHT_FIELDS:
            element = extra_info.find(field)
            if element is None:
                element = ET.SubElement(extra_info, field)
            element.text = value

        tree.write(str(urdf_path), encoding="utf-8", xml_declaration=True)

    def _update_urdf_origins(self, urdf_path: Path) -> None:
        """Append the inverse baked rotation to visual/collision origins."""
        if self._is_identity_rotation:
            return

        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        inverse_rotation = self.rotation.inv().as_matrix()

        for link in root.findall(".//link"):
            for geometry_tag in ("visual", "collision"):
                for geometry in link.findall(geometry_tag):
                    origin = geometry.find("origin")
                    if origin is None:
                        origin = ET.SubElement(
                            geometry,
                            "origin",
                            attrib={"xyz": "0 0 0"},
                        )

                    current_rpy = np.fromstring(
                        origin.get("rpy", "0 0 0"),
                        sep=" ",
                    )
                    if current_rpy.size != 3:
                        raise ValueError(
                            f"Invalid rpy in URDF origin: {urdf_path}"
                        )

                    current_rotation = Rotation.from_euler(
                        "xyz",
                        current_rpy,
                        degrees=False,
                    ).as_matrix()
                    updated_rotation = current_rotation @ inverse_rotation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        updated_rpy = Rotation.from_matrix(
                            updated_rotation
                        ).as_euler("xyz", degrees=False)
                    origin.set(
                        "rpy",
                        " ".join(f"{value:.4f}" for value in updated_rpy),
                    )

        tree.write(str(urdf_path), encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _load_collision_obj(filepath: str) -> list[trimesh.Trimesh]:
        """Robustly load collision OBJ with multiple objects.

        Handles OBJ files with multiple objects/groups by parsing manually
        to avoid issues with trimesh's default loader.

        Args:
            filepath: Path to collision OBJ file.

        Returns:
            List of trimesh objects, one per object group in the file.
        """
        vertices = []
        meshes = []
        current_faces = []

        # Use lazy iteration instead of readlines() for memory efficiency
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                elif line.startswith("f "):
                    parts = line.split()
                    face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    current_faces.append(face)
                elif line.startswith("o ") or line.startswith("g "):
                    if current_faces and vertices:
                        m = trimesh.Trimesh(
                            vertices=vertices,
                            faces=current_faces,
                            process=False,
                        )
                        m.remove_unreferenced_vertices()
                        meshes.append(m)
                    current_faces = []

        # Flush final mesh
        if current_faces and vertices:
            m = trimesh.Trimesh(
                vertices=vertices, faces=current_faces, process=False
            )
            m.remove_unreferenced_vertices()
            meshes.append(m)

        return meshes


def process_asset(
    urdf_path: str | Path,
    scale_factor: float = 1.0,
    rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    keep_urdf_raw_rot: bool = False,
    output_dir: Optional[str | Path] = None,
    inplace: bool = False,
) -> Path:
    """Scale and rotate a complete URDF-based asset."""
    processor = AssetProcessor(
        urdf_path=urdf_path,
        scale_factor=scale_factor,
        rot_xyz=rot_xyz,
        keep_urdf_raw_rot=keep_urdf_raw_rot,
        output_dir=output_dir,
        inplace=inplace,
    )
    return processor.process()


def entrypoint() -> None:
    """Run the asset processing CLI."""
    config = tyro.cli(AssetProcessConfig)
    output_urdf = process_asset(
        urdf_path=config.urdf_path,
        scale_factor=config.scale_factor,
        rot_xyz=config.rot_xyz,
        keep_urdf_raw_rot=config.keep_urdf_raw_rot,
        output_dir=config.output_dir,
        inplace=config.inplace,
    )
    logger.info(f"Processed asset successfully: {output_urdf}")


if __name__ == "__main__":
    entrypoint()
