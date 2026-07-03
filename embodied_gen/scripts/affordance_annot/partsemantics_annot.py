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
import os
import shutil
from dataclasses import dataclass, field
from typing import Literal

import json_repair
import tyro
from embodied_gen.utils.general import filter_warnings

filter_warnings()


from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.io_utils import URDFFile, load_mesh
from embodied_gen.utils.log import logger
from embodied_gen.utils.vis_utils import (
    PALETTE,
    collect_colors,
    render_grid,
    visualize_partsemantics,
)
from embodied_gen.validators.quality_checkers import PartSemanticsChecker

__all__ = [
    "PartSemanticsAnnotConfig",
    "PartSemanticsAnnotator",
    "run_partsemantics_annot",
    "entrypoint",
]


@dataclass
class PartSemanticsAnnotConfig:
    urdf_paths: list[str] = field(default_factory=list)
    output_dirs: list[str] = field(default_factory=list)
    mesh_type: Literal["visual", "collision"] = "visual"
    max_tokens: int = 10240
    max_repairs: int = 2
    visualize: bool = False
    vis_fps: int = 12
    vis_frames_per_part: int = 36
    vis_view_size: int = 1080
    grid_num_images: int = 6
    grid_rows: int = 2
    grid_cols: int = 3
    grid_image_size: int = 512
    debug_mode: bool = False
    overwrite: bool = True


class PartSemanticsAnnotator:
    def __init__(self, cfg: PartSemanticsAnnotConfig):
        self.validate_config(cfg)
        self.cfg = cfg
        self.gpt_client = GPT_CLIENT
        self.checker = PartSemanticsChecker(GPT_CLIENT)
        self.system_prompt = (
            "You are an expert annotator for 3D object affordances. You will receive two aligned multi-view grid images of the same object: "
            "the first grid is the RGB render, and the second grid is a colored part mask where each color denotes one segmented part. "
            "You will also receive the object category and the list of colors that appear in the mask.\n"
            "Your task is to infer affordance information for each meaningful colored part.\n"
            "For every annotated part, provide:\n"
            "1. A concise self-contained part name. Prefer the natural standalone part name when it is clear, e.g. \"lampshade\" or \"light bulb\"; add the object category only when the name would otherwise be ambiguous, e.g. \"stem of apple\" rather than \"stem\".\n"
            "2. Whether the part is realistically useful for robotic grasping with a gripper.\n"
            "3. If it is graspable, concrete robotic grasp scenarios with confidence scores.\n"
            "4. Functional labels: concrete short phrases describing what the part can be used for and what functional role it provides.\n"
            "5. Semantic description: a concise natural-language description of the part, including its role, RGB appearance color, material type, surface finish, tactile texture, shape, relative size/position, and any visible markings when inferable.\n"
            "Use the RGB grid to understand physical appearance and the mask grid to map parts to colors. "
            "The mask color is only an ID for the segmented part; do not describe mask colors as physical object colors. "
            "Only use mask colors from the provided color list. Do not invent mask colors. "
            "If a colored region has no clear affordance, still include it when it is a distinct physical part, with conservative functional labels. "
            "Return only valid JSON. Do not wrap the JSON in Markdown or code fences."
        )
        self.user_prompt = (
            "Object category: {category}\n"
            "Mask color list: {color_names}\n"
            "Image inputs:\n"
            "- Image 1: RGB multi-view grid of the object.\n"
            "- Image 2: part mask multi-view grid aligned with Image 1. Different colors indicate different segmented parts.\n\n"
            "Annotate the affordance of the visible colored parts. Match each part to exactly one color from the color list.\n"
            "For each part, include these fields:\n"
            "- \"part_name\": concise semantic name of the part. Use natural standalone names when they are already clear, such as \"lampshade\", \"light bulb\", \"keyboard\", \"wheel\", or \"mug handle\". Use \"part of object\" only when the standalone part name is ambiguous or too generic, such as \"stem of apple\", \"blade of knife\", or \"base of lamp\".\n"
            "- \"mask_color\": one mask color from the provided color list. This is the segmentation color, not the physical RGB color.\n"
            "- \"graspable\": true or false.\n"
            "- \"grasp_scenarios\": list of dicts, each with \"scenario\" and \"confidence\"; each \"confidence\" belongs to that same scenario and is a float from 0.0 to 1.0 estimating how likely a robot gripper would choose this part for that scenario. Use accurate, distinguishable confidences when multiple parts share the same scenario. Do not give high confidence to theoretically possible but uncommon grasps. Use an empty list if graspable is false.\n"
            "- \"functional_labels\": list of 2 to 6 concrete short phrases describing what this part can be used to do and what functional role it provides.\n"
            "- \"semantic_description\": one concise but complete sentence describing the part's RGB appearance color, material type, surface finish, tactile texture, shape, relative size, location, and any visible markings when inferable. Do not mention the mask color unless it is also the real RGB appearance.\n\n"
            "Guidelines:\n"
            "- Use the RGB grid for appearance and the mask grid for color-to-part mapping.\n"
            "- Treat grasping as robotic gripper grasping, not human hand grasping.\n"
            "- A part is graspable only if it is a plausible and useful place for a robot gripper to hold, pick up, stabilize, pull, or manipulate the object.\n"
            "- If a part is only theoretically graspable but normally poor or uncommon for grasping, keep \"graspable\" True only when there is a realistic scenario, and assign low confidence. For example, an apple stem may be graspable in a delicate stem-picking scenario but is low confidence for lifting the apple.\n"
            "- If a part is functional but not graspable, set \"graspable\" to False and still describe its function.\n"
            "- Functional labels should describe what the part can be used for or what function it enables, such as \"support\", \"contain\", \"cut\", \"press\", \"cover\", \"connect\", \"stabilize\", or \"indicate orientation\".\n"
            "- Semantic descriptions should cover visible material and texture cues when inferable: material type such as metal, plastic, ceramic, rubber, fabric, wood, glass, or food skin; surface finish such as glossy, matte, translucent, reflective, or dull; tactile texture such as soft, hard, smooth, rough, fine-grained, pebbled, ribbed, woven, or granular; inherent patterns or special markings such as wood grain, fabric weave, seams, engraved text, logos, printed icons, labels, or scratches.\n"
            "- Keep semantic descriptions concise, usually one sentence under 35 words, and do not invent material or markings that are not visually supported.\n"
            "- Do not assign affordances to background regions.\n"
            "- Do not include colors that are not in the color list.\n\n"
            "Output format: Return exactly one JSON object with this schema:\n"
            "{{\"affordances\": [{{\"part_name\": \"mug handle\", \"mask_color\": \"Red\", \"graspable\": true, \"grasp_scenarios\": [{{\"scenario\": \"grasp the handle to lift the mug\", \"confidence\": 0.92}}, {{\"scenario\": \"hold the handle to stabilize the mug while pouring\", \"confidence\": 0.84}}], \"functional_labels\": [\"provide a side grip\", \"lift the mug without touching the body\", \"stabilize the mug while pouring\"], \"semantic_description\": \"Curved glossy ceramic handle with a hard smooth surface, matching the mug body and sized for side gripping.\"}}, "
            "{{\"part_name\": \"stem of apple\", \"mask_color\": \"Green\", \"graspable\": true, \"grasp_scenarios\": [{{\"scenario\": \"delicately grasp the stem to orient the apple\", \"confidence\": 0.28}}, {{\"scenario\": \"lift the apple by the stem\", \"confidence\": 0.12}}], \"functional_labels\": [\"connect to branch\", \"indicate top orientation\"], \"semantic_description\": \"Small thin brown woody stem with a hard rough fibrous texture protruding from the apple top.\"}}, "
            "{{\"part_name\": \"body of apple\", \"mask_color\": \"Blue\", \"graspable\": true, \"grasp_scenarios\": [{{\"scenario\": \"grasp the apple body to pick it up\", \"confidence\": 0.88}}], \"functional_labels\": [\"contain edible flesh\", \"support biting\", \"main grasp surface\"], \"semantic_description\": \"Large rounded red apple body with glossy smooth food skin, firm texture, and subtle natural color speckles.\"}}]}}\n"
            "If the target object cannot be identified, output exactly: {{\"affordances\": []}}"
        )

    def validate_config(self, cfg: PartSemanticsAnnotConfig) -> None:
        if not cfg.urdf_paths:
            raise ValueError("urdf_paths must be provided.")

        if len(cfg.output_dirs) == 0:
            cfg.output_dirs = [
                os.path.join(os.path.dirname(path), "affordance")
                for path in cfg.urdf_paths
            ]

        if len(cfg.urdf_paths) != len(cfg.output_dirs):
            raise ValueError(
                "urdf_paths and output_dirs must have the same length, "
                f"got {len(cfg.urdf_paths)} and {len(cfg.output_dirs)}."
            )

        if cfg.grid_num_images != cfg.grid_rows * cfg.grid_cols:
            raise ValueError(
                "grid_num_images must equal grid_rows * grid_cols, "
                f"got {cfg.grid_num_images} and "
                f"{cfg.grid_rows} * {cfg.grid_cols}."
            )

    def check_requires(self, urdf_path: str) -> tuple[str, str]:
        urdf = URDFFile(urdf_path)
        seg_mesh_path = urdf.get_mesh_part_seg_path()
        if not os.path.exists(seg_mesh_path):
            raise FileNotFoundError(
                f"Part segmentation result not found. Run part segmentation before "
                f"partsemantics annotation. Missing: {seg_mesh_path}"
            )
        return seg_mesh_path

    def parse_response(self, response: str) -> dict:
        answer = str(response).strip()
        parsed = json.loads(answer)
        return parsed

    def add_palette_ids(self, payload: dict) -> dict:
        color_to_id = {
            color_name: palette_id for palette_id, _, color_name in PALETTE
        }
        for partsemantics in payload.get("affordances", []):
            if isinstance(partsemantics, dict):
                partsemantics["id"] = color_to_id.get(
                    partsemantics.get("mask_color")
                )
        return payload

    def check_response(
        self, response: str, rgb_grid_path: str, mask_grid_path: str
    ) -> dict:
        passed, message = self.checker(
            response,
            rgb_grid_path,
            mask_grid_path,
        )
        check_info = {
            "success": passed,
            "message": message,
        }
        if not passed:
            feedback = self._parse_checker_message(message)
            modified_response = (
                feedback.get("modified_response")
                if isinstance(feedback, dict)
                else None
            )
            if isinstance(modified_response, str):
                check_info["modified_response"] = modified_response.strip()
            elif modified_response is not None:
                check_info["modified_response"] = json.dumps(
                    modified_response,
                    ensure_ascii=False,
                )
        return check_info

    def _parse_checker_message(self, checker_message: str) -> dict | None:
        if not isinstance(checker_message, str):
            return None

        answer = checker_message.strip()
        try:
            feedback = json_repair.loads(answer)
        except (json.JSONDecodeError, ValueError, TypeError):
            json_start = answer.find("{")
            json_end = answer.rfind("}")
            if json_start < 0 or json_end <= json_start:
                return None
            try:
                feedback = json_repair.loads(answer[json_start : json_end + 1])
            except (json.JSONDecodeError, ValueError, TypeError):
                return None

        return feedback if isinstance(feedback, dict) else None

    def _quality_check_for_save(
        self, quality_check: dict | None
    ) -> dict | None:
        if quality_check is None:
            return None
        if not isinstance(quality_check, dict):
            return {"success": None, "message": str(quality_check)}

        message = quality_check.get("message", "")
        feedback = self._parse_checker_message(message)
        if isinstance(feedback, dict) and feedback.get("reason") is not None:
            message = str(feedback["reason"]).strip()
        elif isinstance(message, str) and message.strip().startswith("NO:"):
            message = message.strip()[3:].strip()

        return {
            "success": quality_check.get("success"),
            "message": message,
        }

    def save_annotation(
        self, response: str, output_dir: str, quality_check: dict | None = None
    ) -> str:
        annotation_path = os.path.join(output_dir, "affordance_annot.json")
        if self.cfg.debug_mode:
            raw_output_path = os.path.join(
                output_dir, "affordance_annot_raw.txt"
            )
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(str(response))

        try:
            payload = self.parse_response(response)
            payload = self.add_palette_ids(payload)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            payload = {
                "status": "parse failed",
                "affordances": [],
                "parse_error": str(exc),
            }

        saved_quality_check = self._quality_check_for_save(quality_check)
        if saved_quality_check is not None:
            payload["quality_check"] = saved_quality_check

        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return annotation_path

    def update_urdf(self, urdf_path: str, annotation_path: str) -> None:
        if urdf_path is None:
            return

        URDFFile(urdf_path).write(
            {
                "custom_data/affordance/affordance_annot": os.path.relpath(
                    annotation_path,
                    os.path.dirname(urdf_path),
                ),
            }
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

    def _process_once(
        self,
        rgb_grid_path: str,
        mask_grid_path: str,
        color_names: str,
        category: str,
    ) -> tuple[bool, dict, str]:
        response = self.gpt_client.query(
            text_prompt=self.user_prompt.format(
                category=category,
                color_names=color_names,
            ),
            image_base64=[rgb_grid_path, mask_grid_path],
            system_role=self.system_prompt,
            params={"max_tokens": self.cfg.max_tokens},
        )
        check_info = {"success": False, "message": "NO: checker was not run"}
        success = False
        max_checks = self.cfg.max_repairs + 1
        for check_idx in range(max_checks):
            logger.info(
                "PartSemantics quality check: attempt "
                f"{check_idx + 1}/{max_checks}"
            )
            check_info = self.check_response(
                response,
                rgb_grid_path,
                mask_grid_path,
            )
            success = check_info.get("success")
            if success:
                return success, check_info, response

            if check_idx == max_checks - 1:
                break

            modified_response = check_info.get("modified_response")
            if modified_response is None:
                break

            response = modified_response
            logger.warning(
                "PartSemantics checker failed; retrying with modified_response. "
            )

        return success, check_info, response

    def process(self, urdf_path: str, output_dir: str) -> bool:
        try:
            return self._process_impl(urdf_path, output_dir)
        except Exception as exc:
            logger.error(
                "PartSemantics annotation failed for URDF {}: {}".format(
                    urdf_path,
                    exc,
                )
            )
            return False

    def _process_impl(self, urdf_path: str, output_dir: str) -> bool:
        annotation_path = os.path.join(output_dir, "affordance_annot.json")
        if not self.cfg.overwrite and os.path.exists(annotation_path):
            logger.info(
                f"Skip existing PartSemantics annotation: {annotation_path}"
            )
            return True

        urdf = URDFFile(urdf_path)
        mesh_path = urdf.get_mesh_path(self.cfg.mesh_type)
        seg_mesh_path = self.check_requires(urdf_path)

        logger.info("Processing PartSemantics annotation...")
        rgb_grid_path, _ = self.render_grid(mesh_path, output_dir)
        mask_grid_path, _ = self.render_grid(
            seg_mesh_path, output_dir, output_subdir="part_seg_renders"
        )

        _, face_ids = load_mesh(
            seg_mesh_path, apply_origin=False, apply_scale=False
        )
        color_names = collect_colors(face_ids)
        logger.info(f"Mask colors: {color_names}")

        category = urdf.get_category()
        success, check_info, response = self._process_once(
            rgb_grid_path,
            mask_grid_path,
            color_names,
            category,
        )
        if success:
            logger.info(
                "PartSemantics annotation passed quality check."
                f"{check_info.get('message')}"
            )
        else:
            logger.error(
                "PartSemantics annotation failed quality check after "
                f"{self.cfg.max_repairs} repair attempts;\nsaving last result. "
            )

        annotation_path = self.save_annotation(
            response, output_dir, check_info
        )
        self.update_urdf(urdf_path, annotation_path)
        if self.cfg.visualize:
            visualize_partsemantics(
                mesh_path,
                seg_mesh_path,
                annotation_path,
                fps=self.cfg.vis_fps,
                frames_per_part=self.cfg.vis_frames_per_part,
                view_size=self.cfg.vis_view_size,
            )

        if not self.cfg.debug_mode:
            for grid_dir in {
                os.path.dirname(rgb_grid_path),
                os.path.dirname(mask_grid_path),
            }:
                if grid_dir:
                    shutil.rmtree(grid_dir, ignore_errors=True)

        return success


def run_partsemantics_annot(cfg: PartSemanticsAnnotConfig) -> None:
    annotator = PartSemanticsAnnotator(cfg)
    for urdf_path, output_dir in zip(
        annotator.cfg.urdf_paths, annotator.cfg.output_dirs
    ):
        annotator.process(urdf_path, output_dir)


def entrypoint() -> None:
    cfg = tyro.cli(PartSemanticsAnnotConfig)
    run_partsemantics_annot(cfg)


if __name__ == "__main__":
    entrypoint()
