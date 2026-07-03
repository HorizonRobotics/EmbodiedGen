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

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

from embodied_gen.utils.general import filter_warnings

filter_warnings()

import numpy as np
import trimesh
import trimesh.transformations as tra
import tyro

sys.path.append(".")
from embodied_gen.utils.geometry import (
    sample_surface_points,
)
from embodied_gen.utils.io_utils import (
    URDFFile,
    load_json,
    load_mesh,
    write_json,
)
from embodied_gen.utils.log import logger
from embodied_gen.utils.vis_utils import show_grasps

__all__ = [
    "GraspGenHyperConfig",
    "GraspGenXHyperConfig",
    "GraspPoseConfig",
    "GraspGenerator",
    "run_grasp",
    "entrypoint",
]

FINGER_CENTER_CACHE: dict[str, np.ndarray] = {
    "franka_panda": np.array([0.0, 0.0, 0.1034], dtype=np.float64),
}

GRIPPER_GEOMETRY_CACHE: dict[str, dict[str, float]] = {
    "franka_panda": {
        "width": 0.10537486,
        "depth": 0.10527314,
    },
}


@dataclass
class GraspGenHyperConfig:
    gripper_config: str = (
        "~/.cache/huggingface/hub/GraspGenModels/checkpoints/graspgen_{}.yml"
    )
    grasp_threshold: float = -1.0
    num_grasps: int = 4000
    return_topk: bool = True
    topk_num_grasps: int = 4000
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    num_sample_points: int = 2000
    min_grasps: int = 40
    max_tries: int = 6
    remove_outliers: bool = False


@dataclass
class GraspGenXHyperConfig:
    checkpoint_root: str = "~/.cache/huggingface/hub/GraspGenXModels/release"
    gen_pth: str | None = None
    dis_pth: str | None = None
    assets_dir: str = (
        "thirdparty/gripper_descriptions/gripper_descriptions/assets"
    )
    grasp_threshold: float = -1.0
    num_grasps: int = 4000
    return_topk: bool = True
    topk_num_grasps: int = 4000
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    num_sample_points: int = 2000
    min_grasps: int = 40
    max_tries: int = 6
    remove_outliers: bool = False


@dataclass
class GraspPoseConfig:
    urdf_paths: list[str] = field(default_factory=list)
    model_name: str = "GraspGen"
    gripper_name: str = "franka_panda"
    mesh_type: Literal["collision", "visual"] = "collision"
    update_urdf: bool = True
    overwrite: bool = True
    gripper_mesh_path: str = (
        f"embodied_gen/scripts/affordance_annot/{gripper_name}/coll_mesh.obj"
    )
    grasps_per_part: int = 20
    visualize_grasps: bool = False
    visualizer_port: int = 8080
    graspgen: GraspGenHyperConfig | None = None


class GraspGenerator:
    def __init__(self, cfg: GraspPoseConfig):
        self.cfg = cfg
        self.grasp_visualizer = None
        self.validate_config(cfg)
        self.modules = self.load_model()
        self.pipeline = self.build_sampler()
        self.finger_center = FINGER_CENTER_CACHE.get(
            self.cfg.gripper_name,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        (
            self.local_contact_points,
            self.local_closing_directions,
        ) = self.build_local_contact_samples()

    def validate_config(self, cfg: GraspPoseConfig) -> None:
        cfg.model_name = cfg.model_name.lower()
        if cfg.model_name not in {"graspgen", "graspgenx"}:
            raise ValueError(
                "unsupported model '{}', available models: GraspGen, GraspGenX".format(
                    cfg.model_name
                )
            )

        cfg.graspgen = (
            GraspGenHyperConfig()
            if cfg.model_name == "graspgen"
            else GraspGenXHyperConfig()
        )
        if cfg.model_name == "graspgen":
            cfg.graspgen.gripper_config = cfg.graspgen.gripper_config.format(
                cfg.gripper_name
            )

    def load_model(self):
        if self.cfg.model_name == "graspgen":
            sys.path.append("thirdparty/GraspGen")
            from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

            return {
                "GraspGenSampler": GraspGenSampler,
                "model_cfg": load_grasp_cfg,
            }
        if self.cfg.model_name == "graspgenx":
            sys.path.append("thirdparty/GraspGenX")
            from graspgenx.grasp_server import GraspGenXSampler
            from graspgenx.utils.checkpoint_io import load_model_cfg

            return {
                "GraspGenSampler": GraspGenXSampler,
                "model_cfg": load_model_cfg,
            }

    def build_sampler(self):
        if self.cfg.model_name == "graspgen":
            gripper_config = os.path.expanduser(
                self.cfg.graspgen.gripper_config
            )

            if not os.path.exists(gripper_config):
                logger.info(
                    "Downloading GraspGen weights from Hugging Face. "
                    "First-time use may take several minutes; please wait patiently."
                )
                from huggingface_hub import snapshot_download

                download_dir = os.path.expanduser(
                    "~/.cache/huggingface/hub/GraspGenModels"
                )
                snapshot_download(
                    repo_id="adithyamurali/GraspGenModels",
                    allow_patterns="checkpoints/*",
                    local_dir=download_dir,
                )

            grasp_cfg = self.modules["model_cfg"](gripper_config)
            return self.modules["GraspGenSampler"](grasp_cfg)
        if self.cfg.model_name == "graspgenx":
            checkpoint_root = os.path.expanduser(
                self.cfg.graspgen.checkpoint_root
            )
            config_path = os.path.join(checkpoint_root, "gen", "config.yaml")
            if not os.path.exists(config_path):
                logger.info(
                    "Downloading GraspGenX weights from Hugging Face. "
                    "First-time use may take several minutes; please wait patiently."
                )
                from huggingface_hub import snapshot_download

                download_dir = os.path.expanduser(
                    "~/.cache/huggingface/hub/GraspGenXModels"
                )
                snapshot_download(
                    repo_id="adithyamurali/GraspGenXModel",
                    allow_patterns="release/*",
                    local_dir=download_dir,
                )
            model_cfg = self.modules["model_cfg"](
                os.path.join(checkpoint_root, "gen"),
                os.path.join(checkpoint_root, "dis"),
                self.cfg.graspgen.gen_pth,
                self.cfg.graspgen.dis_pth,
            )
            return self.modules["GraspGenSampler"](
                model_cfg,
                self.cfg.gripper_name,
                assets_dir=self.cfg.graspgen.assets_dir,
            )

    def update_urdf(self, urdf_path: str | None, grasp_path: str) -> None:
        if urdf_path is None or not self.cfg.update_urdf:
            return

        URDFFile(urdf_path).write(
            {
                "custom_data/affordance/affordance_annot": os.path.relpath(
                    grasp_path,
                    os.path.dirname(urdf_path),
                ),
            }
        )

    def get_part_seg_mesh_transform(self, urdf: URDFFile) -> dict:
        return {
            "origin_xyz": urdf.read(
                ".//custom_data/affordance/visual_seg/origin",
                attr="xyz",
                default="0 0 0",
            ),
            "origin_rpy": urdf.read(
                ".//custom_data/affordance/visual_seg/origin",
                attr="rpy",
                default="0 0 0",
            ),
            "scale": urdf.read(
                ".//custom_data/affordance/visual_seg/geometry/mesh",
                attr="scale",
                default="1.0 1.0 1.0",
            ),
        }

    def add_grasps_to_affordance(
        self,
        affordance_payload: dict,
        grasps: np.ndarray,
        confidences: np.ndarray,
        part_grasp_ids: dict[int, list[int]] | None,
    ) -> None:
        for affordance in affordance_payload.get("affordances", []):
            if (
                not isinstance(affordance, dict)
                or affordance.get("id") is None
            ):
                continue
            affordance.pop("grasps", None)
            affordance.pop("confidences", None)
            grasp_ids = part_grasp_ids.get(int(affordance["id"]), [])
            grasp_group = {}
            for grasp_idx in grasp_ids:
                grasp = grasps[grasp_idx]
                quaternion = tra.quaternion_from_matrix(grasp[:3, :3])
                grasp_group[f"grasp_{grasp_idx}"] = {
                    "confidence": float(confidences[grasp_idx]),
                    "position": grasp[:3, 3].tolist(),
                    "orientation": {
                        "w": float(quaternion[0]),
                        "xyz": quaternion[1:].tolist(),
                    },
                }
            affordance["grasp_group"] = grasp_group

    def infer_grasps(
        self,
        mesh: trimesh.Trimesh,
        mesh_path: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        points = sample_surface_points(
            mesh, self.cfg.graspgen.num_sample_points
        )
        center_transform = tra.translation_matrix(-points.mean(axis=0))
        centered_points = tra.transform_points(
            points, center_transform
        ).astype(np.float32)
        topk_limit = (
            self.cfg.graspgen.topk_num_grasps
            if self.cfg.graspgen.return_topk
            else -1
        )
        grasps, confidences = self.modules["GraspGenSampler"].run_inference(
            centered_points,
            self.pipeline,
            grasp_threshold=self.cfg.graspgen.grasp_threshold,
            num_grasps=self.cfg.graspgen.num_grasps,
            topk_num_grasps=topk_limit,
            min_grasps=self.cfg.graspgen.min_grasps,
            max_tries=self.cfg.graspgen.max_tries,
            remove_outliers=self.cfg.graspgen.remove_outliers,
        )

        confidences = confidences.detach().cpu().numpy()
        grasps = grasps.detach().cpu().numpy()
        if len(grasps) == 0:
            logger.error(f"No grasp poses generated for mesh: {mesh_path}")
            return None

        grasps = np.asarray(
            [tra.inverse_matrix(center_transform) @ grasp for grasp in grasps]
        )
        return grasps, confidences

    def get_grasp_centers(
        self,
        grasps: np.ndarray,
    ) -> np.ndarray:
        local_centers = np.repeat(
            np.asarray(self.finger_center, dtype=np.float64)[None, :],
            len(grasps),
            axis=0,
        )
        grasp_center_points = (
            np.einsum(
                "nij,nj->ni",
                grasps[:, :3, :3],
                local_centers,
            )
            + grasps[:, :3, 3]
        )
        return grasp_center_points

    def build_local_contact_samples(self) -> tuple[np.ndarray, np.ndarray]:
        geometry = GRIPPER_GEOMETRY_CACHE.get(self.cfg.gripper_name)
        if geometry is None:
            return (
                np.asarray(self.finger_center, dtype=np.float64)[None, :],
                np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            )

        half_width = float(geometry["width"]) / 2.0
        depth = float(geometry["depth"])
        z_values = np.linspace(depth - 0.015, depth, 6, dtype=np.float64)
        points = []
        directions = []
        for z_value in z_values:
            points.extend(
                [
                    [-half_width, 0.0, z_value],
                    [half_width, 0.0, z_value],
                ]
            )
            directions.extend(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        return (
            np.asarray(points, dtype=np.float64),
            np.asarray(directions, dtype=np.float64),
        )

    def get_grasp_contact_rays(
        self,
        grasps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_contact_points = np.repeat(
            self.local_contact_points[None, :, :],
            len(grasps),
            axis=0,
        )
        local_closing_directions = np.repeat(
            self.local_closing_directions[None, :, :],
            len(grasps),
            axis=0,
        )
        ray_origins = (
            np.einsum(
                "nij,nkj->nki",
                grasps[:, :3, :3],
                local_contact_points,
            )
            + grasps[:, None, :3, 3]
        )
        ray_directions = np.einsum(
            "nij,nkj->nki",
            grasps[:, :3, :3],
            local_closing_directions,
        )
        ray_norms = np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        ray_directions = np.divide(
            ray_directions,
            ray_norms,
            out=np.zeros_like(ray_directions),
            where=ray_norms > 0.0,
        )
        return ray_origins, ray_directions

    def bind_contact_rays_to_faces(
        self,
        part_mesh: trimesh.Trimesh,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
    ) -> np.ndarray:
        ray_origins = np.asarray(ray_origins, dtype=np.float64)
        ray_directions = np.asarray(ray_directions, dtype=np.float64)

        if (
            not isinstance(part_mesh, trimesh.Trimesh)
            or len(part_mesh.faces) == 0
        ):
            return np.full(len(ray_origins), -1, dtype=np.int64)

        try:
            face_ids = part_mesh.ray.intersects_first(
                ray_origins,
                ray_directions,
            )
            return np.asarray(face_ids, dtype=np.int64)
        except Exception as exc:
            logger.warning(
                "Failed to raycast grasp closing directions to part mesh: "
                f"{exc}"
            )
            return np.full(len(ray_origins), -1, dtype=np.int64)

    def classify_grasps_by_contact(
        self,
        grasps: np.ndarray,
        part_mesh: trimesh.Trimesh,
        face_ids: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if face_ids is None:
            return None, None

        face_ids = np.asarray(face_ids, dtype=np.int64)
        if len(face_ids) != len(part_mesh.faces):
            raise ValueError(
                "face_ids length must match segmentation mesh faces, got "
                f"{len(face_ids)} and {len(part_mesh.faces)}"
            )

        ray_origins, ray_directions = self.get_grasp_contact_rays(grasps)
        num_grasps, num_contact_points, _ = ray_origins.shape
        logger.info(
            "Binding {} grasp closing rays to {} part faces.".format(
                num_grasps * num_contact_points,
                len(part_mesh.faces),
            )
        )
        nearest_face_ids = self.bind_contact_rays_to_faces(
            part_mesh,
            ray_origins.reshape(-1, 3),
            ray_directions.reshape(-1, 3),
        ).reshape(num_grasps, num_contact_points)
        contact_part_ids = self.parts_for_faces(
            nearest_face_ids.reshape(-1),
            face_ids,
        ).reshape(num_grasps, num_contact_points)

        grasp_part_ids = np.full(num_grasps, -1, dtype=np.int64)
        contact_scores = np.zeros(num_grasps, dtype=np.float64)
        for grasp_idx, part_ids in enumerate(contact_part_ids):
            valid_part_ids = part_ids[part_ids >= 0]
            if len(valid_part_ids) == 0:
                continue
            unique_ids, counts = np.unique(
                valid_part_ids,
                return_counts=True,
            )
            best_idx = int(np.argmax(counts))
            grasp_part_ids[grasp_idx] = int(unique_ids[best_idx])
            contact_scores[grasp_idx] = float(counts[best_idx]) / float(
                num_contact_points
            )

        return grasp_part_ids, contact_scores

    def parts_for_faces(
        self,
        nearest_face_ids: np.ndarray,
        face_ids: np.ndarray | None,
    ) -> np.ndarray | None:
        if face_ids is None:
            return None

        nearest_face_ids = np.asarray(nearest_face_ids, dtype=np.int64)
        grasp_part_ids = np.full(len(nearest_face_ids), -1, dtype=np.int64)
        valid_mask = (nearest_face_ids >= 0) & (
            nearest_face_ids < len(face_ids)
        )
        grasp_part_ids[valid_mask] = face_ids[nearest_face_ids[valid_mask]]
        return grasp_part_ids

    def pick_top_grasps(
        self,
        grasps: np.ndarray,
        confidences: np.ndarray,
        part_ids: np.ndarray | None,
        graspable_part_ids: set[int] | None = None,
        part_scores: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]] | None
    ]:
        if part_ids is None:
            return (
                grasps,
                confidences,
                np.arange(len(grasps), dtype=np.int64),
                None,
            )

        part_ids = np.asarray(part_ids, dtype=np.int64)
        selected_indices = []
        selected_part_ids = []
        for part_id in sorted(
            int(pid) for pid in np.unique(part_ids[part_ids >= 0])
        ):
            if (
                graspable_part_ids is not None
                and part_id not in graspable_part_ids
            ):
                continue
            part_indices = np.flatnonzero(part_ids == part_id)
            if part_scores is None:
                rank = np.argsort(
                    -confidences[part_indices],
                    kind="mergesort",
                )
            else:
                rank = np.lexsort(
                    (
                        -confidences[part_indices],
                        -part_scores[part_indices],
                    )
                )
            ranked_indices = part_indices[rank]
            if len(ranked_indices) > self.cfg.grasps_per_part:
                selected = ranked_indices[: self.cfg.grasps_per_part]
            else:
                selected = ranked_indices
            selected_indices.extend(int(idx) for idx in selected)
            selected_part_ids.extend([part_id] * len(selected))

        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if len(selected_indices) == 0:
            return (
                grasps[:0],
                confidences[:0],
                selected_indices,
                {},
            )

        part_grasp_ids: dict[int, list[int]] = {}
        for new_idx, part_id in enumerate(selected_part_ids):
            part_grasp_ids.setdefault(part_id, []).append(new_idx)

        return (
            grasps[selected_indices],
            confidences[selected_indices],
            selected_indices,
            part_grasp_ids,
        )

    def select_grasps(
        self,
        grasps: np.ndarray,
        confidences: np.ndarray,
        part_mesh: trimesh.Trimesh,
        face_ids: np.ndarray,
        affordance_payload: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]]]:
        original_count = len(grasps)

        # Raycast from fingertip samples along the gripper closing direction.
        grasp_center_points = self.get_grasp_centers(grasps)
        grasp_part_ids, contact_scores = self.classify_grasps_by_contact(
            grasps,
            part_mesh,
            face_ids,
        )

        affordances = affordance_payload.get("affordances", [])
        graspable_part_ids = {
            int(affordance["id"])
            for affordance in affordances
            if isinstance(affordance, dict)
            and affordance.get("id") is not None
            and affordance.get("graspable") is True
        }

        # Select top grasps per part
        grasps, confidences, selected_indices, part_grasp_ids = (
            self.pick_top_grasps(
                grasps,
                confidences,
                grasp_part_ids,
                graspable_part_ids,
                contact_scores,
            )
        )
        grasp_center_points = grasp_center_points[selected_indices]
        if len(grasps) == 0:
            logger.info(
                "Selected per-part grasps: {} -> 0 grasps across 0 parts".format(
                    original_count,
                )
            )
            return grasps, confidences, grasp_center_points, part_grasp_ids

        logger.info(
            "Selected per-part grasps: {} -> {} grasps across {} parts, confidences range from {:.4f} to {:.4f}".format(
                original_count,
                len(grasps),
                len(part_grasp_ids),
                np.min(confidences),
                np.max(confidences),
            )
        )
        return grasps, confidences, grasp_center_points, part_grasp_ids

    def process(
        self,
        urdf_path: str,
    ) -> bool:
        try:
            return self._process_impl(urdf_path)
        except Exception as exc:
            logger.error(
                "Grasp generation failed for URDF {}: {}".format(
                    urdf_path,
                    exc,
                )
            )
            return False

    def _process_impl(
        self,
        urdf_path: str,
    ) -> bool:
        urdf = URDFFile(urdf_path)
        mesh_path = urdf.get_mesh_path(self.cfg.mesh_type)
        mesh_transform = urdf.get_mesh_transform(self.cfg.mesh_type)
        seg_mesh_path = urdf.get_mesh_part_seg_path()
        seg_mesh_transform = self.get_part_seg_mesh_transform(urdf)
        affordance_path = urdf.get_affordance_annot_path()
        affordance_payload = load_json(affordance_path)

        if not self.cfg.overwrite:
            has_existing_grasp_group = any(
                isinstance(affordance, dict) and "grasp_group" in affordance
                for affordance in affordance_payload.get("affordances", [])
            )
            if has_existing_grasp_group:
                logger.info(
                    f"Skip existing grasp poses in affordance JSON: {affordance_path}"
                )
                return True

        logger.info(f"Generating grasps for mesh: {mesh_path}")
        mesh = load_mesh(
            mesh_path,
            apply_origin=True,
            apply_scale=True,
            **mesh_transform,
        )

        seg_mesh, face_ids = load_mesh(
            seg_mesh_path,
            apply_origin=True,
            apply_scale=True,
            **seg_mesh_transform,
        )

        grasp_result = self.infer_grasps(mesh, mesh_path)
        if grasp_result is None:
            return False
        grasps, confidences = grasp_result
        grasps, confidences, grasp_center_points, part_grasp_ids = (
            self.select_grasps(
                grasps,
                confidences,
                seg_mesh,
                face_ids,
                affordance_payload,
            )
        )

        self.add_grasps_to_affordance(
            affordance_payload,
            grasps,
            confidences,
            part_grasp_ids,
        )
        write_json(affordance_payload, affordance_path)
        logger.info(f"Saved predicted grasp poses to {affordance_path}")

        self.update_urdf(urdf_path, affordance_path)
        logger.info(
            "Grasp generation succeeded: {} -> {} ({} grasps)".format(
                mesh_path,
                affordance_path,
                len(grasps),
            )
        )
        if self.cfg.visualize_grasps:
            gripper_mesh = load_mesh(self.cfg.gripper_mesh_path)
            self.grasp_visualizer = show_grasps(
                mesh,
                grasps,
                confidences,
                self.cfg.gripper_name,
                self.cfg.model_name,
                grasp_center_points=grasp_center_points,
                visualizer=self.grasp_visualizer,
                visualizer_port=self.cfg.visualizer_port,
                gripper_info=self.pipeline.get_gripper_info()
                if self.cfg.model_name == "graspgenx"
                else None,
                gripper_mesh=gripper_mesh,
            )
        return True


def run_grasp(cfg: GraspPoseConfig) -> None:
    generator = GraspGenerator(cfg)
    for urdf_path in cfg.urdf_paths:
        generator.process(urdf_path)

    if cfg.visualize_grasps and generator.grasp_visualizer is not None:
        logger.info(
            "Keeping grasp visualization alive at http://localhost:{} "
            "(press Ctrl+C to exit).".format(cfg.visualizer_port)
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping grasp visualization.")


def entrypoint(*args, **kwargs) -> None:
    cfg = tyro.cli(GraspPoseConfig)
    run_grasp(cfg)


if __name__ == "__main__":
    entrypoint()
