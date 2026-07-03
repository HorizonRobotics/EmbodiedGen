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
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Literal

import json_repair
from embodied_gen.utils.general import filter_warnings
from embodied_gen.utils.gpt_clients import GPT_CLIENT

filter_warnings()

import numpy as np
import trimesh
import tyro
from huggingface_hub import snapshot_download
from embodied_gen.utils.general import set_seed
from embodied_gen.utils.geometry import (
    EnclosedFaceLabelFillConfig,
    SmoothFaceLabelMergeConfig,
    fill_small_enclosed_face_labels,
    merge_smooth_face_labels,
    normalize_mesh,
)
from embodied_gen.utils.io_utils import URDFFile, load_mesh, save_mesh
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import render_asset3d
from embodied_gen.utils.vis_utils import PALETTE, collect_colors, render_grid
from embodied_gen.validators.quality_checkers import PartSegChecker

__all__ = [
    "FaceIdPostProcessConfig",
    "PartSegConfig",
    "PartSegmenter",
    "entrypoint",
]


@dataclass
class P3SAMConfig:
    threshold: float = 0.95
    post_process: bool = True
    save_mid_res: bool = False
    show_info: bool = True
    is_parallel: bool = False
    seed: int = 42
    clean_mesh_flag: bool = False
    prompt_bs: int = 8  # Lower this for reduced GPU memory usage.


@dataclass
class FaceIdPostProcessConfig:
    enabled: bool = True
    max_smooth_angle_deg: float = 5.0
    vertex_merge_tolerance: float = 0.0
    min_component_area_ratio: float = 0.0
    min_component_faces: int = 1
    preserve_negative: bool = False
    fill_enclosed_components: bool = True
    max_enclosed_component_area_ratio: float = 0.005
    max_enclosed_component_faces: int = 100
    min_enclosure_neighbor_ratio: float = 0.8


@dataclass
class PartSegConfig:
    urdf_paths: list[str] = field(default_factory=list)
    output_dirs: list[str] = field(default_factory=list)
    model_name: str = "p3sam"
    thirdparty_root: str = ""
    mesh_type: Literal["visual", "collision"] = "visual"
    merge_retries: int = 3
    p3sam: P3SAMConfig = field(default_factory=P3SAMConfig)
    face_id_postprocess: FaceIdPostProcessConfig = field(
        default_factory=FaceIdPostProcessConfig
    )
    grid_num_images: int = 6
    grid_rows: int = 2
    grid_cols: int = 3
    grid_image_size: int = 512
    debug_mode: bool = False
    overwrite: bool = True


class PartSegmenter:
    def __init__(self, cfg: PartSegConfig):
        self.validate_config(cfg)
        self.cfg = cfg
        self.seg_checker = PartSegChecker(GPT_CLIENT)
        self.model = self.import_model()
        self.pipeline = self.build_pipeline(self.model)

    def validate_config(self, cfg: PartSegConfig) -> None:
        if cfg.model_name != "p3sam":
            raise ValueError(
                "unsupported model '{}', available models: p3sam".format(
                    cfg.model_name
                )
            )

        if cfg.model_name == "p3sam":
            cfg.thirdparty_root = "thirdparty/Hunyuan3D-Part"

        if not cfg.urdf_paths:
            raise ValueError("urdf_paths must be provided.")

        if len(cfg.output_dirs) == 0:
            cfg.output_dirs = [
                os.path.join(os.path.dirname(path), "affordance")
                for path in cfg.urdf_paths
            ]
        if len(cfg.urdf_paths) != len(cfg.output_dirs):
            raise ValueError(
                "urdf-derived mesh paths and output_dirs must have the same length, "
                f"got {len(cfg.urdf_paths)} and {len(cfg.output_dirs)}."
            )

    def import_model(self):
        if self.cfg.model_name == "p3sam":
            p3sam_root = os.path.join(self.cfg.thirdparty_root, "P3-SAM")
            p3sam_demo = os.path.join(p3sam_root, "demo")
            for path in [p3sam_demo, p3sam_root]:
                if path not in sys.path:
                    sys.path.append(path)
            from embodied_gen.utils.monkey_patch.p3sam import (
                monkey_patch_p3sam,
            )

            monkey_patch_p3sam()
            from auto_mask import AutoMask

            return AutoMask

        raise ValueError("unsupported model '{}'".format(self.cfg.model_name))

    def build_pipeline(self, model):
        if self.cfg.model_name == "p3sam":
            ckpt_path = os.path.expanduser(
                "~/.cache/huggingface/hub/P3-SAM/p3sam/p3sam.safetensors"
            )
            if not os.path.exists(ckpt_path):
                download_dir = os.path.expanduser(
                    "~/.cache/huggingface/hub/P3-SAM"
                )
                logger.info(
                    "Downloading P3-SAM weights from Hugging Face to {}. "
                    "First-time use may take several minutes; please wait patiently.".format(
                        ckpt_path
                    )
                )
                snapshot_download(
                    repo_id="tencent/Hunyuan3D-Part",
                    allow_patterns="p3sam/*",
                    local_dir=download_dir,
                )
            sonata_path = os.path.expanduser(
                "~/.cache/huggingface/hub/sonata/sonata.pth"
            )
            if not os.path.exists(sonata_path):
                logger.info(
                    "Downloading Sonata weights from Hugging Face to {}. "
                    "First-time use may take several minutes; please wait patiently.".format(
                        sonata_path
                    )
                )
            return model(
                ckpt_path=ckpt_path,
                threshold=self.cfg.p3sam.threshold,
                post_process=self.cfg.p3sam.post_process,
            )

        raise ValueError("unsupported model '{}'".format(self.cfg.model_name))

    def build_texture(
        self,
        mesh: trimesh.Trimesh,
        face_colors: np.ndarray,
        min_cell: int = 16,
        max_tex: int = 8192,
    ) -> trimesh.Trimesh:
        if len(face_colors) != len(mesh.faces):
            raise ValueError(
                f"face_colors length must match mesh faces, got {len(face_colors)} and {len(mesh.faces)}"
            )

        from PIL import Image, ImageDraw

        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        n_faces = len(faces)
        grid = int(np.ceil(np.sqrt(n_faces)))
        textured_size = 1 << int(np.ceil(np.log2(max(1, grid * min_cell))))
        textured_size = min(max_tex, max(min_cell, textured_size))
        cell = textured_size / grid

        image = Image.new(
            "RGBA", (textured_size, textured_size), (255, 255, 255, 255)
        )
        draw = ImageDraw.Draw(image)
        uv = np.zeros((n_faces * 3, 2), dtype=np.float64)
        local_uv = np.asarray(
            [
                [0.18, 0.18],
                [0.82, 0.18],
                [0.18, 0.82],
            ],
            dtype=np.float64,
        )

        for face_idx, color in enumerate(face_colors):
            row, col = divmod(face_idx, grid)
            rgba = tuple(int(channel) for channel in color)
            x0 = int(np.floor(col * cell))
            y0 = int(np.floor(row * cell))
            x1 = int(np.ceil((col + 1) * cell))
            y1 = int(np.ceil((row + 1) * cell))
            draw.rectangle((x0, y0, x1, y1), fill=rgba)
            uv[face_idx * 3 : face_idx * 3 + 3, 0] = (
                col + local_uv[:, 0]
            ) / grid
            uv[face_idx * 3 : face_idx * 3 + 3, 1] = (
                1.0 - (row + local_uv[:, 1]) / grid
            )

        textured_mesh = trimesh.Trimesh(
            vertices=vertices[faces.reshape(-1)],
            faces=np.arange(n_faces * 3, dtype=np.int64).reshape(n_faces, 3),
            process=False,
        )
        material = trimesh.visual.texture.SimpleMaterial(
            image=image,
            diffuse=[255, 255, 255, 255],
            name="part_segmentation",
        )
        textured_mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uv, material=material
        )
        return textured_mesh

    def remap_to_palette(self, face_ids: np.ndarray) -> np.ndarray:
        face_ids = np.asarray(face_ids)
        mapped_ids = np.full(face_ids.shape, -1, dtype=np.int64)
        unique_ids = np.unique(face_ids[face_ids >= 0])
        if len(unique_ids) > len(PALETTE):
            return None

        for palette_idx, part_id in enumerate(unique_ids):
            palette_id = PALETTE[palette_idx][0]
            mapped_ids[face_ids == part_id] = palette_id
        return mapped_ids

    def export_seg_mesh(
        self,
        mesh,
        face_ids: np.ndarray,
        output_dir: str,
        mesh_transform: dict | None = None,
    ) -> None:
        output_path = os.path.join(output_dir, "mesh_part_seg.glb")
        palette_colors = {
            palette_id: color for palette_id, color, _ in PALETTE
        }

        face_colors = np.zeros((len(face_ids), 4), dtype=np.uint8)
        face_colors[:, 3] = 255
        for idx, part_id in enumerate(face_ids):
            if part_id < 0:
                face_colors[idx, :3] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                face_colors[idx, :3] = palette_colors[int(part_id)]

        color_mesh = self.build_texture(mesh.copy(), face_colors)
        color_mesh.metadata["face_ids"] = face_ids.tolist()
        if mesh_transform is None:
            save_mesh(
                color_mesh, output_path, apply_origin=False, apply_scale=False
            )
        else:
            save_mesh(
                color_mesh,
                output_path,
                apply_origin=True,
                apply_scale=True,
                **mesh_transform,
            )
        return output_path

    def count_parts(self, face_ids: np.ndarray) -> int:
        valid_ids = np.asarray(face_ids)[np.asarray(face_ids) >= 0]
        return int(len(np.unique(valid_ids)))

    def postprocess_face_ids(
        self,
        mesh: trimesh.Trimesh,
        face_ids: np.ndarray | None,
    ) -> np.ndarray | None:
        if face_ids is None or not self.cfg.face_id_postprocess.enabled:
            return face_ids

        merge_cfg = SmoothFaceLabelMergeConfig(
            max_smooth_angle_deg=self.cfg.face_id_postprocess.max_smooth_angle_deg,
            vertex_merge_tolerance=self.cfg.face_id_postprocess.vertex_merge_tolerance,
            min_component_area_ratio=self.cfg.face_id_postprocess.min_component_area_ratio,
            min_component_faces=self.cfg.face_id_postprocess.min_component_faces,
            preserve_negative=self.cfg.face_id_postprocess.preserve_negative,
        )
        face_ids, stats = merge_smooth_face_labels(mesh, face_ids, merge_cfg)
        if stats["faces_changed"] > 0:
            logger.info(
                "Part seg face_id postprocess changed "
                f"{stats['faces_changed']} faces across "
                f"{stats['components_changed']} smooth components."
            )
        if self.cfg.face_id_postprocess.fill_enclosed_components:
            fill_cfg = EnclosedFaceLabelFillConfig(
                max_component_area_ratio=(
                    self.cfg.face_id_postprocess.max_enclosed_component_area_ratio
                ),
                max_component_faces=(
                    self.cfg.face_id_postprocess.max_enclosed_component_faces
                ),
                min_enclosure_neighbor_ratio=(
                    self.cfg.face_id_postprocess.min_enclosure_neighbor_ratio
                ),
                vertex_merge_tolerance=(
                    self.cfg.face_id_postprocess.vertex_merge_tolerance
                ),
            )
            face_ids, stats = fill_small_enclosed_face_labels(
                mesh,
                face_ids,
                fill_cfg,
            )
            if stats["faces_changed"] > 0:
                logger.info(
                    "Part seg face_id postprocess filled "
                    f"{stats['faces_changed']} enclosed faces across "
                    f"{stats['components_filled']} components."
                )
        return face_ids

    def update_urdf(
        self,
        urdf_path: str,
        seg_mesh_path: str,
        n_parts: int,
        check_info: str,
        mesh_transform: dict | None = None,
    ) -> None:
        urdf_dir = os.path.dirname(urdf_path)
        seg_mesh_file = (
            os.path.relpath(
                seg_mesh_path,
                urdf_dir,
            )
            if seg_mesh_path
            else None
        )
        origin_attrs = {"xyz": "0 0 0", "rpy": "0 0 0"}
        mesh_attrs = {"filename": seg_mesh_file, "scale": "1 1 1"}
        if mesh_transform:
            origin_attrs = {
                "xyz": str(mesh_transform["origin_xyz"]),
                "rpy": str(mesh_transform["origin_rpy"]),
            }
            mesh_attrs["scale"] = str(mesh_transform["scale"])

        updates = {
            "custom_data/affordance/part_count": n_parts,
            "custom_data/affordance/visual_seg": {
                "text": None,
                "clear_attrs": True,
                "clear_children": True,
            },
            "custom_data/affordance/visual_seg/origin": {
                "attrs": origin_attrs,
            },
            "custom_data/affordance/visual_seg/geometry/mesh": {
                "attrs": mesh_attrs,
            },
            "custom_data/quality/PartSegChecker": check_info,
        }

        URDFFile(urdf_path).write(updates)

    def render_video(
        self, mesh_path: str, output_dir: str, output_subdir: str = "renders"
    ) -> list[str]:
        return render_asset3d(
            mesh_path,
            output_root=output_dir,
            distance=5.0,
            num_images=128,
            elevation=(45.0,),
            gen_color_mp4=True,
            output_subdir=output_subdir,
            pbr_light_factor=1.5,
            pbr_metallic=False,
        )

    def render_grid(
        self,
        mesh_path: str,
        output_dir: str,
        output_subdir: str = "renders",
    ) -> tuple[str, list[str]]:
        grid_path, view_paths = render_grid(
            mesh_path,
            output_dir,
            output_subdir=output_subdir,
            num_images=self.cfg.grid_num_images,
            grid_rows=self.cfg.grid_rows,
            grid_cols=self.cfg.grid_cols,
            view_size=self.cfg.grid_image_size,
        )
        return grid_path, view_paths

    def render_seg_result(
        self,
        mesh: trimesh.Trimesh,
        face_ids: np.ndarray,
        output_dir: str,
        mesh_transform: dict | None = None,
    ) -> tuple[str, str, str]:
        seg_mesh_path = self.export_seg_mesh(
            mesh,
            face_ids,
            output_dir,
            mesh_transform,
        )
        mask_grid_path, _ = self.render_grid(
            seg_mesh_path,
            output_dir,
            output_subdir="part_seg_renders",
        )
        color_names = collect_colors(face_ids)
        logger.info(f"Part seg face colors: {color_names}")
        return seg_mesh_path, mask_grid_path, color_names

    def apply_merge_hints(
        self,
        face_ids: np.ndarray,
        check_info: str,
    ) -> tuple[np.ndarray, bool]:
        merge_groups = self.parse_merge_groups(check_info)
        if not merge_groups:
            return face_ids, False

        merged_face_ids = np.asarray(face_ids).copy()
        color_to_id = {
            color_name.lower(): palette_id
            for palette_id, _, color_name in PALETTE
        }
        applied_groups = 0

        for group in merge_groups:
            color_names = group.get("colors", [])
            palette_ids = []
            for color_name in color_names:
                palette_id = color_to_id.get(str(color_name).strip().lower())
                if palette_id is not None and palette_id not in palette_ids:
                    palette_ids.append(palette_id)

            active_ids = [
                palette_id
                for palette_id in palette_ids
                if np.any(merged_face_ids == palette_id)
            ]
            if len(active_ids) < 2:
                continue

            target_id = max(
                active_ids,
                key=lambda palette_id: int(
                    np.count_nonzero(merged_face_ids == palette_id)
                ),
            )
            changed_faces = 0
            for palette_id in active_ids:
                if palette_id == target_id:
                    continue
                source_mask = merged_face_ids == palette_id
                changed_faces += int(np.count_nonzero(source_mask))
                merged_face_ids[source_mask] = target_id

            applied_groups += 1
            logger.info(
                "Applied PartSegChecker merge group for %s: colors=%s, "
                "target_palette_id=%s, faces_changed=%s",
                group.get("target_part", "unknown part"),
                color_names,
                target_id,
                changed_faces,
            )

        return merged_face_ids, applied_groups > 0

    def parse_merge_groups(self, check_info: str) -> list[dict]:
        if not isinstance(check_info, str):
            return []

        json_start = check_info.find("{")
        json_end = check_info.rfind("}")
        if json_start < 0 or json_end <= json_start:
            return []

        try:
            parsed = json_repair.loads(check_info[json_start : json_end + 1])
        except Exception as exc:
            logger.warning(
                f"Failed to parse PartSegChecker merge hints: {exc}"
            )
            return []

        merge_groups = parsed.get("merge_groups", [])
        if not isinstance(merge_groups, list):
            return []

        valid_groups = []
        for group in merge_groups:
            if not isinstance(group, dict):
                continue
            colors = group.get("colors", [])
            if not isinstance(colors, list) or len(colors) < 2:
                continue
            valid_groups.append(group)
        return valid_groups

    def infer_seg(self, mesh: trimesh.Trimesh, output_dir: str) -> bool:
        unit_mesh = normalize_mesh(mesh)
        if self.cfg.model_name == "p3sam":
            set_seed(self.cfg.p3sam.seed)
            os.makedirs(output_dir, exist_ok=True)
            _, face_ids, _ = self.pipeline.predict_aabb(
                unit_mesh,
                save_path=output_dir,
                **asdict(self.cfg.p3sam),
            )
        else:
            raise ValueError(
                "unsupported model '{}'".format(self.cfg.model_name)
            )
        face_ids = self.remap_to_palette(face_ids)
        return face_ids

    def process(self, urdf_path: str, output_dir: str) -> bool:
        try:
            return self._process_impl(urdf_path, output_dir)
        except Exception as exc:
            logger.error(
                "Part segmentation failed for URDF {}: {}".format(
                    urdf_path,
                    exc,
                )
            )
            return False

    def _process_impl(self, urdf_path: str, output_dir: str) -> bool:
        seg_mesh_path = os.path.join(output_dir, "mesh_part_seg.glb")
        mask_grid_path = None
        if not self.cfg.overwrite and os.path.exists(seg_mesh_path):
            logger.info(
                f"Skip existing part segmentation result: {seg_mesh_path}"
            )
            return True

        urdf = URDFFile(urdf_path)
        category = urdf.get_category()
        mesh_path = urdf.get_mesh_path(self.cfg.mesh_type)
        mesh_transform = {}
        if urdf_path is not None:
            mesh_transform = urdf.get_mesh_transform(self.cfg.mesh_type)
        mesh = load_mesh(
            mesh_path,
            apply_origin=True,
            apply_scale=True,
            **mesh_transform,
        )
        rgb_grid_path, _ = self.render_grid(mesh_path, output_dir)

        face_ids = self.infer_seg(mesh, output_dir)
        face_ids = self.postprocess_face_ids(mesh, face_ids)

        if face_ids is None:
            logger.error("Part seg failed due to exceeding palette size. ")
            success, check_info = (
                False,
                f"NO: part number exceeds palette size {len(PALETTE)}",
            )
            seg_mesh_path = None
        else:
            seg_mesh_path, mask_grid_path, color_names = (
                self.render_seg_result(
                    mesh,
                    face_ids,
                    output_dir,
                    mesh_transform if urdf_path is not None else None,
                )
            )
            success, check_info = self.seg_checker(
                category,
                color_names,
                rgb_grid_path,
                mask_grid_path,
            )
            self.retry_count = 0
            while not success and self.retry_count < self.cfg.merge_retries:
                merged_face_ids, merge_applied = self.apply_merge_hints(
                    face_ids,
                    check_info,
                )
                if merge_applied:
                    face_ids = merged_face_ids
                    # 备份mask_grid_path
                    mask_grid_path_bak = mask_grid_path.replace(
                        "affordance_grid",
                        f"affordance_grid_{self.retry_count}",
                    )
                    if os.path.exists(mask_grid_path):
                        shutil.copy2(mask_grid_path, mask_grid_path_bak)
                    seg_mesh_path, mask_grid_path, color_names = (
                        self.render_seg_result(
                            mesh,
                            face_ids,
                            output_dir,
                            mesh_transform if urdf_path is not None else None,
                        )
                    )
                    success, check_info = self.seg_checker(
                        category,
                        color_names,
                        rgb_grid_path,
                        mask_grid_path,
                    )
                self.retry_count += 1

        if success:
            logger.info(
                f"Part seg successful for {mesh_path} with {self.count_parts(face_ids)} parts. "
            )
        else:
            logger.error(
                f"Part seg failed for {mesh_path}. Checker info: {check_info}"
            )
        self.update_urdf(
            urdf_path,
            seg_mesh_path,
            self.count_parts(face_ids) if face_ids is not None else 0,
            check_info,
            mesh_transform if urdf_path is not None else None,
        )
        if self.cfg.debug_mode:
            self.render_video(mesh_path, output_dir)
            if seg_mesh_path:
                self.render_video(
                    seg_mesh_path, output_dir, output_subdir="part_seg_renders"
                )
        else:
            grid_dirs = {os.path.dirname(rgb_grid_path)}
            if mask_grid_path:
                grid_dirs.add(os.path.dirname(mask_grid_path))
            for grid_dir in grid_dirs:
                if grid_dir:
                    shutil.rmtree(grid_dir, ignore_errors=True)
        return success


def run_part_seg(cfg: PartSegConfig) -> None:
    segmenter = PartSegmenter(cfg)
    for urdf_path, output_dir in zip(cfg.urdf_paths, cfg.output_dirs):
        segmenter.process(urdf_path, output_dir)


def entrypoint(*args, **kwargs) -> None:
    cfg = tyro.cli(PartSegConfig)
    run_part_seg(cfg)


if __name__ == "__main__":
    entrypoint()
