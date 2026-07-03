import json
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import trimesh
import trimesh.transformations as tra

if TYPE_CHECKING:
    from embodied_gen.utils.geometry import MeshInfo

__all__ = [
    "URDFFile",
    "load_json",
    "load_mesh",
    "load_mesh_info",
    "save_mesh",
    "write_json",
]

DEFAULT_URDF_ORIGIN_XYZ = (0.0, 0.0, 0.0)
DEFAULT_URDF_ORIGIN_RPY = (1.5708, 0.0, 0.0)
DEFAULT_URDF_SCALE = (1.0, 1.0, 1.0)


def load_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload


def write_json(payload: dict, path: str | os.PathLike) -> None:
    output_path = os.fspath(path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def _sapien_pose_from_mesh_transform(mesh_transform: Mapping[str, str]):
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation as R

    origin_xyz = [
        float(value) for value in mesh_transform["origin_xyz"].split()
    ]
    origin_rpy = [
        float(value) for value in mesh_transform["origin_rpy"].split()
    ]

    rotation = R.from_euler("xyz", origin_rpy)
    quat_xyzw = rotation.as_quat()
    local_pose = sapien.Pose(
        p=origin_xyz,
        q=[
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        ],
    )
    return local_pose


def _mesh_scale_from_transform(
    mesh_transform: Mapping[str, str],
) -> np.ndarray:
    return np.array(
        [float(v) for v in mesh_transform["scale"].split()],
        dtype=np.float64,
    )


def load_mesh_info(urdf_path: str | os.PathLike) -> "MeshInfo":
    from embodied_gen.utils.geometry import MeshInfo

    urdf = URDFFile(urdf_path)
    collision_mesh_path = urdf.get_mesh_path("collision")
    collision_mesh_transform = urdf.get_mesh_transform("collision")
    collision_mesh_scale = _mesh_scale_from_transform(collision_mesh_transform)
    collision_local_pose = _sapien_pose_from_mesh_transform(
        collision_mesh_transform
    )

    try:
        visual_mesh_path = urdf.get_mesh_path("visual")
        visual_mesh_transform = urdf.get_mesh_transform("visual")
    except ValueError:
        visual_mesh_path = collision_mesh_path
        visual_mesh_transform = collision_mesh_transform
    visual_mesh_scale = _mesh_scale_from_transform(visual_mesh_transform)
    visual_local_pose = _sapien_pose_from_mesh_transform(visual_mesh_transform)

    mesh = load_mesh(
        collision_mesh_path,
        apply_origin=True,
        **collision_mesh_transform,
    )
    object_height = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])
    static_friction, dynamic_friction = urdf.get_collision_friction()

    return MeshInfo(
        actor_name=urdf.get_robot_name(),
        collision_mesh_path=collision_mesh_path,
        collision_mesh_scale=collision_mesh_scale,
        collision_local_pose=collision_local_pose,
        visual_mesh_path=visual_mesh_path,
        visual_mesh_scale=visual_mesh_scale,
        visual_local_pose=visual_local_pose,
        transformed_mesh=mesh,
        object_height=object_height,
        mass=urdf.get_mass(),
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
    )


class URDFFile:
    """Small XML helper for reading and writing fields inside one URDF file."""

    def __init__(self, urdf_path: str | os.PathLike):
        self.urdf_path = os.fspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)
        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()

    def reload(self) -> None:
        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()

    def save(
        self,
        urdf_path: str | os.PathLike | None = None,
        *,
        indent: bool = True,
        indent_space: str = "   ",
    ) -> None:
        output_path = (
            os.fspath(urdf_path) if urdf_path is not None else self.urdf_path
        )
        if indent:
            ET.indent(self.tree, space=indent_space)
        self.tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def read(
        self,
        path: str,
        *,
        attr: str | None = None,
        default: Any = None,
        required: bool = False,
        all_matches: bool = False,
        strip: bool = True,
    ) -> Any:
        nodes = (
            self.root.findall(path) if all_matches else [self.root.find(path)]
        )
        if not all_matches and nodes[0] is None:
            if required:
                raise ValueError(
                    f"URDF path not found: {path} in {self.urdf_path}"
                )
            return default

        values = [
            self._read_node_value(
                node, attr=attr, default=default, strip=strip
            )
            for node in nodes
            if node is not None
        ]
        if all_matches:
            if required and not values:
                raise ValueError(
                    f"URDF path not found: {path} in {self.urdf_path}"
                )
            return values
        return values[0]

    def get_mesh_path(self, mesh_type: str = "visual") -> str:
        mesh_path = self.read(
            f".//{mesh_type}/geometry/mesh",
            attr="filename",
            required=True,
        )
        return self._resolve_path(mesh_path)

    def get_robot_name(self, default: str = "dropped_object") -> str:
        return self.read(".", attr="name", default=default)

    def get_category(self) -> str:
        category = self.read(".//extra_info/category", required=True)
        if not category:
            raise ValueError(f"Empty category in {self.urdf_path}")
        return category

    def get_mesh_part_seg_path(self) -> str:
        mesh_path = self.read(
            ".//custom_data/affordance/visual_seg/geometry/mesh",
            attr="filename",
            required=True,
        )
        return self._resolve_path(mesh_path)

    def get_affordance_annot_path(self) -> str:
        annot_path = self.read(
            ".//custom_data/affordance/affordance_annot",
            required=True,
        )
        return self._resolve_path(annot_path)

    def get_mesh_transform(self, mesh_type: str = "visual") -> dict:
        return {
            "origin_xyz": self.read(
                f".//{mesh_type}/origin",
                attr="xyz",
                default="0 0 0",
            ),
            "origin_rpy": self.read(
                f".//{mesh_type}/origin",
                attr="rpy",
                default="0 0 0",
            ),
            "scale": self.read(
                f".//{mesh_type}/geometry/mesh",
                attr="scale",
                default="1 1 1",
            ),
        }

    def get_mass(self) -> float | None:
        mass = self.read(".//inertial/mass", attr="value")
        return None if mass is None else float(mass)

    def get_collision_friction(
        self,
        *,
        default_static: float = 0.7,
        default_dynamic: float = 0.6,
    ) -> tuple[float, float]:
        static_friction = self.read(
            ".//collision/gazebo/mu1",
            default=str(default_static),
        )
        dynamic_friction = self.read(
            ".//collision/gazebo/mu2",
            default=str(default_dynamic),
        )
        return float(static_friction), float(dynamic_friction)

    def get_prismatic_joint_control_info(self) -> dict[str, float]:
        lower_limits = []
        upper_limits = []
        effort_limits = []
        damping_values = []
        for joint_node in self.root.findall(".//joint"):
            if joint_node.get("type") != "prismatic":
                continue
            limit_node = joint_node.find("limit")
            if limit_node is None:
                continue
            lower_limits.append(float(limit_node.get("lower", "0.0")))
            upper_limits.append(float(limit_node.get("upper", "0.0")))
            effort_limits.append(float(limit_node.get("effort", "0.0")))
            dynamics_node = joint_node.find("dynamics")
            if dynamics_node is not None:
                damping_values.append(
                    float(dynamics_node.get("damping", "0.0"))
                )

        if not lower_limits or not upper_limits:
            raise ValueError(
                f"No prismatic joint limits found in {self.urdf_path}"
            )

        valid_efforts = [effort for effort in effort_limits if effort > 0.0]
        valid_damping = [
            damping for damping in damping_values if damping > 0.0
        ]
        return {
            "open_qpos": min(upper_limits),
            "close_qpos": max(lower_limits),
            "drive_damping": max(valid_damping) if valid_damping else 0.0,
            "force_limit": min(valid_efforts) if valid_efforts else 0.0,
        }

    def get_link_names(self) -> list[str]:
        return [
            link_name
            for link_name in (
                link_node.get("name")
                for link_node in self.root.findall("link")
            )
            if link_name
        ]

    def get_child_link_names(
        self,
        *,
        name_contains: str | None = None,
    ) -> list[str]:
        child_link_names = []
        name_query = (
            name_contains.lower() if name_contains is not None else None
        )
        for joint_node in self.root.findall("joint"):
            child_node = joint_node.find("child")
            link_name = (
                child_node.get("link") if child_node is not None else None
            )
            if not link_name:
                continue
            if name_query is not None and name_query not in link_name.lower():
                continue
            child_link_names.append(link_name)
        return list(dict.fromkeys(child_link_names))

    def get_link_transforms(self) -> dict[str, np.ndarray]:
        child_to_joint = {}
        for joint_node in self.root.findall("joint"):
            child_node = joint_node.find("child")
            parent_node = joint_node.find("parent")
            child_name = (
                child_node.get("link") if child_node is not None else None
            )
            parent_name = (
                parent_node.get("link") if parent_node is not None else None
            )
            if child_name and parent_name:
                child_to_joint[child_name] = joint_node

        transforms: dict[str, np.ndarray] = {}

        def link_transform(link_name: str) -> np.ndarray:
            if link_name in transforms:
                return transforms[link_name]
            joint_node = child_to_joint.get(link_name)
            if joint_node is None:
                transforms[link_name] = np.eye(4, dtype=np.float64)
                return transforms[link_name]

            parent_name = joint_node.find("parent").get("link")
            transforms[link_name] = link_transform(
                parent_name
            ) @ self._joint_transform(joint_node)
            return transforms[link_name]

        for link_name in self.get_link_names():
            link_transform(link_name)
        return transforms

    def load_link_geometry_mesh(
        self,
        link_name: str,
        geometry_type: Literal["collision", "visual"] = "collision",
    ) -> trimesh.Trimesh:
        link_node = next(
            (
                node
                for node in self.root.findall("link")
                if node.get("name") == link_name
            ),
            None,
        )
        if link_node is None:
            raise ValueError(
                f"URDF link not found: {link_name} in {self.urdf_path}"
            )

        geom_node = link_node.find(geometry_type)
        if geom_node is None and geometry_type == "collision":
            geom_node = link_node.find("visual")
        if geom_node is None:
            raise ValueError(
                f"link {link_name} does not contain {geometry_type} geometry"
            )

        mesh_node = geom_node.find("geometry/mesh")
        mesh_filename = (
            mesh_node.get("filename") if mesh_node is not None else None
        )
        if mesh_filename is None:
            raise ValueError(
                f"link {link_name} does not contain a mesh geometry"
            )

        origin_node = geom_node.find("origin")
        origin_xyz = (
            origin_node.get("xyz", "0 0 0")
            if origin_node is not None
            else None
        )
        origin_rpy = (
            origin_node.get("rpy", "0 0 0")
            if origin_node is not None
            else None
        )
        return load_mesh(
            self._resolve_path(mesh_filename),
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            scale=mesh_node.get("scale", "1.0 1.0 1.0"),
            apply_origin=True,
            apply_scale=True,
        )

    def write(
        self,
        updates: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        save: bool = True,
        urdf_path: str | os.PathLike | None = None,
        indent: bool = True,
        indent_space: str = "   ",
    ) -> None:
        for update in self._normalize_updates(updates):
            node = self._find_or_create(update["path"])
            if update.get("clear_attrs", False):
                node.attrib.clear()
            if update.get("clear_children", False):
                node.clear()
            if "text" in update:
                node.text = (
                    None if update["text"] is None else str(update["text"])
                )
            for key, value in update.get("attrs", {}).items():
                node.set(key, str(value))

        if save:
            self.save(urdf_path, indent=indent, indent_space=indent_space)

    @staticmethod
    def _read_node_value(
        node: ET.Element,
        *,
        attr: str | None,
        default: Any,
        strip: bool,
    ) -> Any:
        value = node.get(attr) if attr is not None else node.text
        if value is None:
            return default
        if strip and isinstance(value, str):
            return value.strip()
        return value

    def _resolve_path(self, path: str | os.PathLike) -> str:
        path = os.fspath(path)
        if not os.path.isabs(path):
            path = os.path.join(self.urdf_dir, path)
        return os.path.normpath(path)

    @staticmethod
    def _origin_transform(origin_node: ET.Element | None) -> np.ndarray:
        if origin_node is None:
            return np.eye(4, dtype=np.float64)

        xyz = _parse_xyz_rpy(
            origin_node.get("xyz", "0 0 0"), (0.0, 0.0, 0.0), "origin xyz"
        )
        rpy = _parse_xyz_rpy(
            origin_node.get("rpy", "0 0 0"), (0.0, 0.0, 0.0), "origin rpy"
        )
        transform = tra.euler_matrix(*rpy, axes="sxyz")
        transform[:3, 3] = xyz
        return transform.astype(np.float64, copy=False)

    @staticmethod
    def _joint_axis(joint_node: ET.Element) -> np.ndarray:
        axis_node = joint_node.find("axis")
        axis = np.asarray(
            _parse_xyz_rpy(
                axis_node.get("xyz", "1 0 0")
                if axis_node is not None
                else None,
                (1.0, 0.0, 0.0),
                "joint axis",
            ),
            dtype=np.float64,
        )
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            raise ValueError(
                f"joint axis must be non-zero, got {axis.tolist()}"
            )
        return axis / norm

    @staticmethod
    def _joint_default_position(joint_node: ET.Element) -> float:
        if joint_node.get("type") != "prismatic":
            return 0.0

        limit_node = joint_node.find("limit")
        if limit_node is None:
            return 0.0
        return float(limit_node.get("upper", limit_node.get("lower", "0.0")))

    @classmethod
    def _joint_transform(cls, joint_node: ET.Element) -> np.ndarray:
        transform = cls._origin_transform(joint_node.find("origin"))
        joint_position = cls._joint_default_position(joint_node)
        if joint_position == 0.0:
            return transform

        motion = np.eye(4, dtype=np.float64)
        motion[:3, 3] = cls._joint_axis(joint_node) * joint_position
        return transform @ motion

    @staticmethod
    def _normalize_updates(
        updates: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(updates, Mapping):
            normalized = []
            for path, value in updates.items():
                if isinstance(value, Mapping) and any(
                    key in value
                    for key in (
                        "text",
                        "attrs",
                        "clear_attrs",
                        "clear_children",
                    )
                ):
                    item = {"path": path, **dict(value)}
                else:
                    item = {"path": path, "text": value}
                normalized.append(item)
            return normalized

        normalized = []
        for update in updates:
            if "path" not in update:
                raise ValueError(f"URDF write update missing path: {update}")
            normalized.append(dict(update))
        return normalized

    def _find_or_create(self, path: str) -> ET.Element:
        existing = self.root.find(path)
        if existing is not None:
            return existing

        node = self.root
        for tag in self._creation_path_parts(path):
            child = node.find(tag)
            if child is None:
                child = ET.SubElement(node, tag)
            node = child
        return node

    def _creation_path_parts(self, path: str) -> list[str]:
        clean_path = path.strip()
        if clean_path in ("", "."):
            return []
        if clean_path.startswith(".//"):
            clean_path = clean_path[3:]
        elif clean_path.startswith("./"):
            clean_path = clean_path[2:]
        if clean_path.startswith(f"{self.root.tag}/"):
            clean_path = clean_path[len(self.root.tag) + 1 :]

        parts = [part for part in clean_path.split("/") if part]
        if any(
            any(token in part for token in ("[", "]", "@", "*"))
            for part in parts
        ):
            raise ValueError(
                f"URDF write can only create simple element paths, got: {path}"
            )
        return parts


def _normalize_scale(scale) -> tuple[float, float, float]:
    if isinstance(scale, str):
        values = [float(value) for value in scale.split()]
    elif isinstance(scale, (int, float, np.number)):
        values = [float(scale)]
    else:
        values = [float(value) for value in scale]

    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"scale must have 1 or 3 values, got {scale}")
    return tuple(values)


def _parse_xyz_rpy(
    values: Iterable[float] | str | None,
    default: tuple[float, float, float],
    name: str,
) -> list[float]:
    if values is None:
        parsed = list(default)
    elif isinstance(values, str):
        parsed = [float(value) for value in values.split()]
    else:
        parsed = [float(value) for value in values]

    if len(parsed) != 3:
        raise ValueError(f"{name} must have 3 values, got {parsed}")
    return parsed


def _apply_mesh_scale(
    mesh: trimesh.Trimesh,
    scale: tuple[float, float, float] | str | Iterable[float] | float,
) -> None:
    scale_array = np.asarray(_normalize_scale(scale), dtype=np.float64)
    if np.allclose(scale_array, scale_array[0]):
        mesh.apply_scale(float(scale_array[0]))
        return

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(scale_array)
    mesh.apply_transform(transform)


def _apply_inverse_mesh_scale(
    mesh: trimesh.Trimesh,
    scale: tuple[float, float, float] | str | Iterable[float] | float,
) -> None:
    scale_array = np.asarray(_normalize_scale(scale), dtype=np.float64)
    if np.any(np.isclose(scale_array, 0.0)):
        raise ValueError(f"scale must be non-zero to invert, got {scale}")

    inverse_scale = 1.0 / scale_array
    if np.allclose(inverse_scale, inverse_scale[0]):
        mesh.apply_scale(float(inverse_scale[0]))
        return

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(inverse_scale)
    mesh.apply_transform(transform)


def _apply_origin_transform(
    mesh: trimesh.Trimesh,
    origin_xyz: Iterable[float] | str | None,
    origin_rpy: Iterable[float] | str | None,
) -> None:
    xyz = _parse_xyz_rpy(origin_xyz, (0.0, 0.0, 0.0), "origin_xyz")
    rpy = _parse_xyz_rpy(origin_rpy, (0.0, 0.0, 0.0), "origin_rpy")
    transform = tra.euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    mesh.apply_transform(transform.astype(np.float64, copy=False))


def _apply_inverse_origin_transform(
    mesh: trimesh.Trimesh,
    origin_xyz: Iterable[float] | str | None,
    origin_rpy: Iterable[float] | str | None,
) -> None:
    xyz = _parse_xyz_rpy(origin_xyz, (0.0, 0.0, 0.0), "origin_xyz")
    rpy = _parse_xyz_rpy(origin_rpy, (0.0, 0.0, 0.0), "origin_rpy")
    transform = tra.euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    mesh.apply_transform(
        np.linalg.inv(transform).astype(np.float64, copy=False)
    )


def load_mesh(
    mesh_path: str | os.PathLike,
    *,
    origin_xyz: Iterable[float] | str | None = None,
    origin_rpy: Iterable[float] | str | None = None,
    scale: tuple[float, float, float]
    | str
    | Iterable[float]
    | float
    | None = None,
    apply_origin: bool = True,
    apply_scale: bool = True,
) -> trimesh.Trimesh:
    """Load a mesh and optionally apply URDF mesh scale and origin transform."""

    mesh = trimesh.load(os.fspath(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"mesh is not a triangle mesh: {mesh_path}")

    if apply_scale and scale is not None:
        _apply_mesh_scale(mesh, scale)
    if apply_origin:
        _apply_origin_transform(mesh, origin_xyz, origin_rpy)

    if "face_ids" in mesh.metadata:
        return mesh, np.asarray(mesh.metadata["face_ids"], dtype=np.int64)
    return mesh


def save_mesh(
    mesh: trimesh.Trimesh,
    output_path: str | os.PathLike,
    *,
    origin_xyz: Iterable[float] | str | None = DEFAULT_URDF_ORIGIN_XYZ,
    origin_rpy: Iterable[float] | str | None = DEFAULT_URDF_ORIGIN_RPY,
    scale: tuple[float, float, float]
    | str
    | Iterable[float]
    | float
    | None = DEFAULT_URDF_SCALE,
    apply_origin: bool = True,
    apply_scale: bool = True,
    copy: bool = True,
) -> str:
    """Save a mesh by optionally undoing the default URDF scale and origin."""

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"mesh must be a trimesh.Trimesh, got {type(mesh).__name__}"
        )

    mesh_to_save = mesh.copy() if copy else mesh
    if apply_origin:
        _apply_inverse_origin_transform(mesh_to_save, origin_xyz, origin_rpy)
    if apply_scale and scale is not None:
        _apply_inverse_mesh_scale(mesh_to_save, scale)

    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    mesh_to_save.export(output_path)
    return output_path
