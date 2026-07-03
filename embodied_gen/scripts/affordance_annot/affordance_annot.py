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

import dataclasses
import os
import sys
import time
from dataclasses import dataclass, field

import tyro

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from embodied_gen.utils.general import filter_warnings

filter_warnings()

from embodied_gen.scripts.affordance_annot.eval_grasps import (
    EvalGraspsConfig,
    run_eval,
)
from embodied_gen.scripts.affordance_annot.gen_grasp import (
    GraspGenerator,
    GraspPoseConfig,
)
from embodied_gen.scripts.affordance_annot.part_seg import (
    PartSegConfig,
    PartSegmenter,
)
from embodied_gen.scripts.affordance_annot.partsemantics_annot import (
    PartSemanticsAnnotator,
    PartSemanticsAnnotConfig,
)
from embodied_gen.utils.log import logger

__all__ = [
    "AffordanceAnnotConfig",
    "AffordanceAnnotator",
    "run_affordance_annot",
    "entrypoint",
]


@dataclass
class AffordanceAnnotConfig:
    urdf_paths: list[str] = field(default_factory=list)
    output_dirs: list[str] = field(default_factory=list)
    run_part_seg: bool = True
    run_partsemantics_annot: bool = True
    run_grasp: bool = True
    run_grasp_eval: bool = True
    part_seg: PartSegConfig = field(default_factory=PartSegConfig)
    partsemantics: PartSemanticsAnnotConfig = field(
        default_factory=PartSemanticsAnnotConfig
    )
    grasp: GraspPoseConfig = field(default_factory=GraspPoseConfig)
    grasp_eval: EvalGraspsConfig = field(default_factory=EvalGraspsConfig)


class AffordanceAnnotator:
    def __init__(self, cfg: AffordanceAnnotConfig):
        self.validate_config(cfg)
        self.cfg = cfg
        self.part_segmenter: PartSegmenter | None = None
        self.partsemantics_annotator: PartSemanticsAnnotator | None = None
        self.grasp_generator: GraspGenerator | None = None
        self.asset_timings: list[dict[str, float | str]] = []

    def validate_config(self, cfg: AffordanceAnnotConfig) -> None:
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

    def sync_batch_stage_config(self, cfg):
        updates = {"urdf_paths": list(self.cfg.urdf_paths)}
        if "output_dirs" in {field.name for field in dataclasses.fields(cfg)}:
            updates["output_dirs"] = list(self.cfg.output_dirs)
        return dataclasses.replace(cfg, **updates)

    def sync_single_stage_config(self, cfg, urdf_path: str, output_dir: str):
        updates = {"urdf_paths": [urdf_path]}
        if "output_dirs" in {field.name for field in dataclasses.fields(cfg)}:
            updates["output_dirs"] = [output_dir]
        return dataclasses.replace(cfg, **updates)

    def process_part_seg(self, urdf_path: str, output_dir: str) -> bool:
        if not self.cfg.run_part_seg:
            logger.info("Skipping part segmentation stage.")
            return True

        return bool(self.get_part_segmenter().process(urdf_path, output_dir))

    def process_partsemantics_annot(
        self, urdf_path: str, output_dir: str
    ) -> bool:
        if not self.cfg.run_partsemantics_annot:
            logger.info("Skipping part semantics annotation stage.")
            return True

        return bool(
            self.get_partsemantics_annotator().process(urdf_path, output_dir)
        )

    def process_grasp(self, urdf_path: str, output_dir: str) -> bool:
        if not self.cfg.run_grasp:
            logger.info("Skipping grasp generation stage.")
            return True

        return bool(self.get_grasp_generator().process(urdf_path))

    def process_grasp_eval(self, urdf_path: str, output_dir: str) -> bool:
        if not self.cfg.run_grasp_eval:
            logger.info("Skipping grasp evaluation stage.")
            return True

        cfg = self.sync_single_stage_config(
            self.cfg.grasp_eval,
            urdf_path,
            output_dir,
        )
        run_eval(cfg)
        return True

    def get_part_segmenter(self) -> PartSegmenter:
        if self.part_segmenter is None:
            cfg = self.sync_batch_stage_config(self.cfg.part_seg)
            self.part_segmenter = PartSegmenter(cfg)
        return self.part_segmenter

    def get_partsemantics_annotator(self) -> PartSemanticsAnnotator:
        if self.partsemantics_annotator is None:
            cfg = self.sync_batch_stage_config(self.cfg.partsemantics)
            self.partsemantics_annotator = PartSemanticsAnnotator(cfg)
        return self.partsemantics_annotator

    def get_grasp_generator(self) -> GraspGenerator:
        if self.grasp_generator is None:
            cfg = self.sync_batch_stage_config(self.cfg.grasp)
            self.grasp_generator = GraspGenerator(cfg)
        return self.grasp_generator

    def run_stage(
        self,
        stage_name: str,
        stage_fn,
        urdf_path: str,
    ) -> bool:
        try:
            success = bool(stage_fn())
        except Exception as exc:
            logger.warning(
                "{} failed for URDF {}: {}".format(
                    stage_name,
                    urdf_path,
                    exc,
                )
            )
            return False

        if not success:
            logger.warning(
                "{} returned failure for URDF {}".format(
                    stage_name,
                    urdf_path,
                )
            )
        return success

    def process_one_urdf(
        self,
        urdf_path: str,
        output_dir: str,
        index: int,
    ) -> dict[str, bool]:
        start_time = time.perf_counter()
        logger.info(
            "Starting affordance pipeline for URDF {}/{}: {}".format(
                index,
                len(self.cfg.urdf_paths),
                urdf_path,
            )
        )
        stages = [
            (
                "part segmentation",
                lambda: self.process_part_seg(urdf_path, output_dir),
            ),
            (
                "part semantics annotation",
                lambda: self.process_partsemantics_annot(urdf_path, output_dir),
            ),
            (
                "grasp generation",
                lambda: self.process_grasp(urdf_path, output_dir),
            ),
            (
                "grasp evaluation",
                lambda: self.process_grasp_eval(urdf_path, output_dir),
            ),
        ]

        results = {}
        for stage_name, stage_fn in stages:
            logger.info(f"Starting affordance stage: {stage_name}")
            results[stage_name] = self.run_stage(
                stage_name,
                stage_fn,
                urdf_path,
            )
            logger.info(
                "Finished affordance stage: {} ({})".format(
                    stage_name,
                    "success" if results[stage_name] else "failed",
                )
            )

        failed_stages = [
            stage_name for stage_name, success in results.items() if not success
        ]
        if failed_stages:
            logger.warning(
                "Affordance pipeline finished with failed stages for {}: {}".format(
                    urdf_path,
                    ", ".join(failed_stages),
                )
            )
        else:
            logger.info(f"Affordance pipeline finished successfully: {urdf_path}")

        elapsed_seconds = time.perf_counter() - start_time
        self.asset_timings.append(
            {
                "urdf_path": urdf_path,
                "output_dir": output_dir,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        logger.info(
            "Affordance pipeline time for {}: {:.2f}s".format(
                urdf_path,
                elapsed_seconds,
            )
        )
        return results

    def log_timing_summary(self) -> None:
        if not self.asset_timings:
            return

        total_seconds = sum(
            float(item["elapsed_seconds"]) for item in self.asset_timings
        )
        asset_count = len(self.asset_timings)
        logger.info(
            "Affordance pipeline total time: {:.2f}s for {} asset(s).".format(
                total_seconds,
                asset_count,
            )
        )
        logger.info(
            "Affordance pipeline average time per asset: {:.2f}s".format(
                total_seconds / asset_count,
            )
        )

    def process(self) -> list[dict[str, bool]]:
        results = []
        for idx, (urdf_path, output_dir) in enumerate(
            zip(self.cfg.urdf_paths, self.cfg.output_dirs),
            start=1,
        ):
            results.append(self.process_one_urdf(urdf_path, output_dir, idx))
        self.log_timing_summary()
        return results


def run_affordance_annot(cfg: AffordanceAnnotConfig) -> None:
    annotator = AffordanceAnnotator(cfg)
    annotator.process()


def entrypoint(*args, **kwargs) -> None:
    cfg = tyro.cli(AffordanceAnnotConfig)
    run_affordance_annot(cfg)


if __name__ == "__main__":
    entrypoint()
