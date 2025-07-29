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
import random
from collections import defaultdict, deque
from functools import wraps

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.log import logger

__all__ = [
    "bfs_placement",
    "with_seed",
    "matrix_to_pose",
    "pose_to_matrix",
]


def matrix_to_pose(matrix: np.ndarray) -> list[float]:
    """Convert a 4x4 transformation matrix to a pose (x, y, z, qx, qy, qz, qw).

    Args:
        matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        List[float]: Pose as [x, y, z, qx, qy, qz, qw].
    """
    x, y, z = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    quat = R.from_matrix(rot_mat).as_quat()
    qx, qy, qz, qw = quat

    return [x, y, z, qx, qy, qz, qw]


def pose_to_matrix(pose: list[float]) -> np.ndarray:
    """Convert pose (x, y, z, qx, qy, qz, qw) to a 4x4 transformation matrix.

    Args:
        List[float]: Pose as [x, y, z, qx, qy, qz, qw].

    Returns:
        matrix (np.ndarray): 4x4 transformation matrix.
    """
    x, y, z, qx, qy, qz, qw = pose
    r = R.from_quat([qx, qy, qz, qw])
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    matrix[:3, 3] = [x, y, z]

    return matrix


def compute_xy_bbox(
    vertices: np.ndarray, col_x: int = 0, col_y: int = 2
) -> list[float]:
    x_vals = vertices[:, col_x]
    y_vals = vertices[:, col_y]
    return x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()


def has_iou_conflict(
    new_box: list[float],
    placed_boxes: list[list[float]],
    iou_threshold: float = 0.0,
) -> bool:
    new_min_x, new_max_x, new_min_y, new_max_y = new_box
    for min_x, max_x, min_y, max_y in placed_boxes:
        ix1 = max(new_min_x, min_x)
        iy1 = max(new_min_y, min_y)
        ix2 = min(new_max_x, max_x)
        iy2 = min(new_max_y, max_y)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter_area > iou_threshold:
            return True
    return False


def with_seed(seed_attr_name: str = "seed"):
    """A parameterized decorator that temporarily sets the random seed."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            seed = kwargs.get(seed_attr_name, None)
            if seed is not None:
                py_state = random.getstate()
                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                try:
                    result = func(*args, **kwargs)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    torch.set_rng_state(torch_state)
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


@with_seed("seed")
def bfs_placement(
    layout_info: LayoutInfo,
    floor_margin: float = 0,
    max_attempts: int = 1000,
    seed: int = None,
) -> LayoutInfo:
    object_mapping = layout_info.objs_mapping

    root = list(layout_info.tree.keys())[0]
    queue = deque([((root, None), layout_info.tree.get(root, []))])
    position = {}  # node: [x, y, z, qx, qy, qz, qw]
    default_quat = [0, 0, 0, 1]
    parent_bbox_xy = {}
    placed_boxes_map = defaultdict(list)

    while queue:
        (node, relation), children = queue.popleft()
        if node not in object_mapping:
            continue

        if object_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
            position[node] = [0, 0, -floor_margin, *default_quat]
        else:
            try:
                mesh_path = f"{layout_info.assets[node]}/mesh/{node.replace(' ', '_')}.obj"
                mesh = trimesh.load(mesh_path)
            except Exception as e:
                logger.error(f"{node} mesh loading failed: {e}, skip.")
                continue

            z1 = np.percentile(mesh.vertices[:, 1], 1)
            z2 = np.percentile(mesh.vertices[:, 1], 99)
            x1, x2, y1, y2 = compute_xy_bbox(mesh.vertices)

            if object_mapping[node] == Scene3DItemEnum.CONTEXT.value:
                position[node] = [0, 0, -round(z1, 4), *default_quat]
                parent_bbox_xy[node] = [x1, x2, y1, y2, z1, z2]
            elif object_mapping[node] in [
                Scene3DItemEnum.MANIPULATED_OBJS.value,
                Scene3DItemEnum.DISTRACTOR_OBJS.value,
            ]:
                parent_node = next(
                    (
                        p
                        for p, c in layout_info.tree.items()
                        if any(cc[0] == node for cc in c)
                    ),
                    None,
                )
                if parent_node is None:
                    continue

                parent_pos = position[parent_node]
                (
                    p_x1,
                    p_x2,
                    p_y1,
                    p_y2,
                    p_z1,
                    p_z2,
                ) = parent_bbox_xy[parent_node]

                obj_dx = x2 - x1
                obj_dy = y2 - y1

                for _ in range(max_attempts):
                    node_x1 = random.uniform(
                        p_x1 + obj_dx * 1.0, p_x2 - obj_dx * 1.0
                    )
                    node_y1 = random.uniform(
                        p_y1 + obj_dy * 1.0, p_y2 - obj_dy * 1.0
                    )
                    node_box = [
                        node_x1,
                        node_x1 + obj_dx,
                        node_y1,
                        node_y1 + obj_dy,
                    ]
                    z_offset = 0
                    if not has_iou_conflict(
                        node_box, placed_boxes_map[parent_node]
                    ):
                        break
                else:
                    logger.warning(
                        f"Cannot place {node} on {parent_node} without overlap"
                        f" after {max_attempts} attempts, place beside {parent_node}."
                    )
                    beside_margin = 0.1
                    for _ in range(max_attempts):
                        node_x1 = random.choice(
                            [
                                random.uniform(
                                    p_x1 - obj_dx - beside_margin,
                                    p_x1 - obj_dx,
                                ),
                                random.uniform(p_x2, p_x2 + beside_margin),
                            ]
                        )
                        node_y1 = random.choice(
                            [
                                random.uniform(
                                    p_y1 - obj_dy - beside_margin,
                                    p_y1 - obj_dy,
                                ),
                                random.uniform(p_y2, p_y2 + beside_margin),
                            ]
                        )
                        node_box = [
                            node_x1,
                            node_x1 + obj_dx,
                            node_y1,
                            node_y1 + obj_dy,
                        ]
                        z_offset = -(parent_pos[2] + p_z2)
                        if not has_iou_conflict(
                            node_box, placed_boxes_map[parent_node]
                        ):
                            break

                placed_boxes_map[parent_node].append(node_box)

                abs_cx = parent_pos[0] + node_box[0] + obj_dx / 2
                abs_cy = parent_pos[1] + node_box[2] + obj_dy / 2
                abs_cz = parent_pos[2] + p_z2 - z1 + z_offset
                position[node] = [
                    round(v, 4)
                    for v in [abs_cx, abs_cy, abs_cz, *default_quat]
                ]
                parent_bbox_xy[node] = [x1, x2, y1, y2, z1, z2]

        for child, relation in children:
            queue.append(((child, relation), layout_info.tree.get(child, [])))

    layout_info.position = position

    return layout_info


def compose_mesh_scene(
    layout_info: LayoutInfo, out_scene_path: str, with_bg: bool = False
) -> None:
    object_mapping = Scene3DItemEnum.object_mapping(layout_info.relation)
    scene = trimesh.Scene()
    for node in layout_info.assets:
        if object_mapping[node] == Scene3DItemEnum.BACKGROUND.value:
            mesh_path = f"{layout_info.assets[node]}/mesh_model.ply"
            if not with_bg:
                continue
        else:
            mesh_path = (
                f"{layout_info.assets[node]}/mesh/{node.replace(' ', '_')}.obj"
            )

        mesh = trimesh.load(mesh_path)
        offset = np.array(layout_info.position[node])[[0, 2, 1]]
        mesh.vertices += offset
        scene.add_geometry(mesh, node_name=node)

    os.makedirs(os.path.dirname(out_scene_path), exist_ok=True)
    scene.export(out_scene_path)
    logger.info(f"Composed interactive 3D layout saved in {out_scene_path}")

    return
