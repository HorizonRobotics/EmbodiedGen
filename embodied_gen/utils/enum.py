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

from dataclasses import dataclass, field
from enum import Enum

from dataclasses_json import DataClassJsonMixin

__all__ = [
    "RenderItems",
    "Scene3DItemEnum",
    "SpatialRelationEnum",
    "RobotItemEnum",
]


@dataclass
class RenderItems(str, Enum):
    IMAGE = "image_color"
    ALPHA = "image_mask"
    VIEW_NORMAL = "image_view_normal"
    GLOBAL_NORMAL = "image_global_normal"
    POSITION_MAP = "image_position"
    DEPTH = "image_depth"
    ALBEDO = "image_albedo"
    DIFFUSE = "image_diffuse"


@dataclass
class Scene3DItemEnum(str, Enum):
    BACKGROUND = "background"
    CONTEXT = "context"
    ROBOT = "robot"
    MANIPULATED_OBJS = "manipulated_objs"
    DISTRACTOR_OBJS = "distractor_objs"
    OTHERS = "others"

    @classmethod
    def object_list(cls, layout_relation: dict) -> list:
        return (
            [
                layout_relation[cls.BACKGROUND.value],
                layout_relation[cls.CONTEXT.value],
            ]
            + layout_relation[cls.MANIPULATED_OBJS.value]
            + layout_relation[cls.DISTRACTOR_OBJS.value]
        )

    @classmethod
    def object_mapping(cls, layout_relation):
        relation_mapping = {
            # layout_relation[cls.ROBOT.value]: cls.ROBOT.value,
            layout_relation[cls.BACKGROUND.value]: cls.BACKGROUND.value,
            layout_relation[cls.CONTEXT.value]: cls.CONTEXT.value,
        }
        relation_mapping.update(
            {
                item: cls.MANIPULATED_OBJS.value
                for item in layout_relation[cls.MANIPULATED_OBJS.value]
            }
        )
        relation_mapping.update(
            {
                item: cls.DISTRACTOR_OBJS.value
                for item in layout_relation[cls.DISTRACTOR_OBJS.value]
            }
        )

        return relation_mapping


@dataclass
class SpatialRelationEnum(str, Enum):
    ON = "ON"  # objects on the table
    IN = "IN"  # objects in the room
    INSIDE = "INSIDE"  # objects inside the shelf/rack
    FLOOR = "FLOOR"  # object floor room/bin


@dataclass
class RobotItemEnum(str, Enum):
    FRANKA = "franka"
    UR5 = "ur5"
    PIPER = "piper"


@dataclass
class LayoutInfo(DataClassJsonMixin):
    tree: dict[str, list]
    relation: dict[str, str | list[str]]
    objs_desc: dict[str, str] = field(default_factory=dict)
    objs_mapping: dict[str, str] = field(default_factory=dict)
    assets: dict[str, str] = field(default_factory=dict)
    quality: dict[str, str] = field(default_factory=dict)
    position: dict[str, list[float]] = field(default_factory=dict)
