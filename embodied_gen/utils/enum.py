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
    "LayoutInfo",
    "AssetType",
    "SimAssetMapper",
]


@dataclass
class RenderItems(str, Enum):
    """Enumeration of render item types for 3D scenes.

    Attributes:
        IMAGE: Color image.
        ALPHA: Mask image.
        VIEW_NORMAL: View-space normal image.
        GLOBAL_NORMAL: World-space normal image.
        POSITION_MAP: Position map image.
        DEPTH: Depth image.
        ALBEDO: Albedo image.
        DIFFUSE: Diffuse image.
    """

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
    """Enumeration of 3D scene item categories.

    Attributes:
        BACKGROUND: Background objects.
        CONTEXT: Contextual objects.
        ROBOT: Robot entity.
        MANIPULATED_OBJS: Objects manipulated by the robot.
        DISTRACTOR_OBJS: Distractor objects.
        OTHERS: Other objects.

    Methods:
        object_list(layout_relation): Returns a list of objects in the scene.
        object_mapping(layout_relation): Returns a mapping from object to category.
    """

    BACKGROUND = "background"
    CONTEXT = "context"
    ROBOT = "robot"
    MANIPULATED_OBJS = "manipulated_objs"
    DISTRACTOR_OBJS = "distractor_objs"
    OTHERS = "others"

    @classmethod
    def object_list(cls, layout_relation: dict) -> list:
        """Returns a list of objects in the scene.

        Args:
            layout_relation: Dictionary mapping categories to objects.

        Returns:
            List of objects in the scene.
        """
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
        """Returns a mapping from object to category.

        Args:
            layout_relation: Dictionary mapping categories to objects.

        Returns:
            Dictionary mapping object names to their category.
        """
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
    """Enumeration of spatial relations for objects in a scene.

    Attributes:
        ON: Objects on a surface (e.g., table).
        IN: Objects in a container or room.
        INSIDE: Objects inside a shelf or rack.
        FLOOR: Objects on the floor.
    """

    ON = "ON"  # objects on the table
    IN = "IN"  # objects in the room
    INSIDE = "INSIDE"  # objects inside the shelf/rack
    FLOOR = "FLOOR"  # object floor room/bin


@dataclass
class RobotItemEnum(str, Enum):
    """Enumeration of supported robot types.

    Attributes:
        FRANKA: Franka robot.
        UR5: UR5 robot.
        PIPER: Piper robot.
    """

    FRANKA = "franka"
    UR5 = "ur5"
    PIPER = "piper"


@dataclass
class LayoutInfo(DataClassJsonMixin):
    """Data structure for layout information in a 3D scene.

    Attributes:
        tree: Hierarchical structure of scene objects.
        relation: Spatial relations between objects.
        objs_desc: Descriptions of objects.
        objs_mapping: Mapping from object names to categories.
        assets: Asset file paths for objects.
        quality: Quality information for assets.
        position: Position coordinates for objects.
    """

    tree: dict[str, list]
    relation: dict[str, str | list[str]]
    objs_desc: dict[str, str] = field(default_factory=dict)
    objs_mapping: dict[str, str] = field(default_factory=dict)
    assets: dict[str, str] = field(default_factory=dict)
    quality: dict[str, str] = field(default_factory=dict)
    position: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class AssetType(str):
    """Enumeration for asset types.

    Supported types:
        MJCF: MuJoCo XML format.
        USD: Universal Scene Description format.
        URDF: Unified Robot Description Format.
        MESH: Mesh file format.
    """

    MJCF = "mjcf"
    USD = "usd"
    URDF = "urdf"
    MESH = "mesh"


class SimAssetMapper:
    """Maps simulator names to asset types.

    Provides a mapping from simulator names to their corresponding asset type.

    Example:
        ```py
        from embodied_gen.utils.enum import SimAssetMapper
        asset_type = SimAssetMapper["isaacsim"]
        print(asset_type)  # Output: 'usd'
        ```

    Methods:
        __class_getitem__(key): Returns the asset type for a given simulator name.
    """

    _mapping = dict(
        ISAACSIM=AssetType.USD,
        ISAACGYM=AssetType.URDF,
        MUJOCO=AssetType.MJCF,
        GENESIS=AssetType.MJCF,
        SAPIEN=AssetType.URDF,
        PYBULLET=AssetType.URDF,
    )

    @classmethod
    def __class_getitem__(cls, key: str):
        """Returns the asset type for a given simulator name.

        Args:
            key: Name of the simulator.

        Returns:
            AssetType corresponding to the simulator.

        Raises:
            KeyError: If the simulator name is not recognized.
        """
        key = key.upper()
        if key.startswith("SAPIEN"):
            key = "SAPIEN"
        return cls._mapping[key]
