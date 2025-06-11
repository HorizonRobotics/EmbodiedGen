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


import logging
from typing import Tuple, Union

import igraph
import numpy as np
import pyvista as pv
import spaces
import torch
import utils3d
from pymeshfix import _meshfix
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


__all__ = ["MeshFixer"]


def _radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def _halton_sequence(dim, n):
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    return [_radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def _hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + _halton_sequence(dim - 1, n)


def _sphere_hammersley_seq(n, num_samples, offset=(0, 0), remap=False):
    """Generate a point on a unit sphere using the Hammersley sequence.

    Args:
        n (int): The index of the sample.
        num_samples (int): The total number of samples.
        offset (tuple, optional): Offset for the u and v coordinates.
        remap (bool, optional): Whether to remap the u coordinate.

    Returns:
        list: A list containing the spherical coordinates [phi, theta].
    """
    u, v = _hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]

    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3

    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


class MeshFixer(object):
    """MeshFixer simplifies and repairs 3D triangle meshes by TSDF.

    Attributes:
        vertices (torch.Tensor): A tensor of shape (V, 3) representing vertex positions.
        faces (torch.Tensor): A tensor of shape (F, 3) representing face indices.
        device (str): Device to run computations on, typically "cuda" or "cpu".

    Main logic reference: https://github.com/microsoft/TRELLIS/blob/main/trellis/utils/postprocessing_utils.py#L22
    """

    def __init__(
        self,
        vertices: Union[torch.Tensor, np.ndarray],
        faces: Union[torch.Tensor, np.ndarray],
        device: str = "cuda",
    ) -> None:
        self.device = device
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices)
        self.vertices = vertices

        if isinstance(faces, np.ndarray):
            faces = torch.tensor(faces)
        self.faces = faces

    @staticmethod
    def log_mesh_changes(method):
        def wrapper(self, *args, **kwargs):
            logger.info(
                f"Before {method.__name__}: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} faces"  # noqa
            )
            result = method(self, *args, **kwargs)
            logger.info(
                f"After {method.__name__}: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} faces"  # noqa
            )
            return result

        return wrapper

    @log_mesh_changes
    def fill_holes(
        self,
        max_hole_size: float,
        max_hole_nbe: int,
        resolution: int,
        num_views: int,
        norm_mesh_ratio: float = 1.0,
    ) -> None:
        self.vertices = self.vertices * norm_mesh_ratio
        vertices, self.faces = self._fill_holes(
            self.vertices,
            self.faces,
            max_hole_size,
            max_hole_nbe,
            resolution,
            num_views,
        )
        self.vertices = vertices / norm_mesh_ratio

    @staticmethod
    @torch.no_grad()
    def _fill_holes(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        max_hole_size: float,
        max_hole_nbe: int,
        resolution: int,
        num_views: int,
    ) -> Union[torch.Tensor, torch.Tensor]:
        yaws, pitchs = [], []
        for i in range(num_views):
            y, p = _sphere_hammersley_seq(i, num_views)
            yaws.append(y)
            pitchs.append(p)

        yaws, pitchs = torch.tensor(yaws).to(vertices), torch.tensor(
            pitchs
        ).to(vertices)
        radius, fov = 2.0, torch.deg2rad(torch.tensor(40)).to(vertices)
        projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)

        views = []
        for yaw, pitch in zip(yaws, pitchs):
            orig = (
                torch.tensor(
                    [
                        torch.sin(yaw) * torch.cos(pitch),
                        torch.cos(yaw) * torch.cos(pitch),
                        torch.sin(pitch),
                    ]
                ).to(vertices)
                * radius
            )
            view = utils3d.torch.view_look_at(
                orig,
                torch.tensor([0, 0, 0]).to(vertices),
                torch.tensor([0, 0, 1]).to(vertices),
            )
            views.append(view)
        views = torch.stack(views, dim=0)

        # Rasterize the mesh
        visibility = torch.zeros(
            faces.shape[0], dtype=torch.int32, device=faces.device
        )
        rastctx = utils3d.torch.RastContext(backend="cuda")

        for i in tqdm(
            range(views.shape[0]), total=views.shape[0], desc="Rasterizing"
        ):
            view = views[i]
            buffers = utils3d.torch.rasterize_triangle_faces(
                rastctx,
                vertices[None],
                faces,
                resolution,
                resolution,
                view=view,
                projection=projection,
            )
            face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
            face_id = torch.unique(face_id).long()
            visibility[face_id] += 1

        # Normalize visibility by the number of views
        visibility = visibility.float() / num_views

        # Mincut: Identify outer and inner faces
        edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
        boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
        connected_components = utils3d.torch.compute_connected_components(
            faces, edges, face2edge
        )

        outer_face_indices = torch.zeros(
            faces.shape[0], dtype=torch.bool, device=faces.device
        )
        for i in range(len(connected_components)):
            outer_face_indices[connected_components[i]] = visibility[
                connected_components[i]
            ] > min(
                max(
                    visibility[connected_components[i]].quantile(0.75).item(),
                    0.25,
                ),
                0.5,
            )

        outer_face_indices = outer_face_indices.nonzero().reshape(-1)
        inner_face_indices = torch.nonzero(visibility == 0).reshape(-1)

        if inner_face_indices.shape[0] == 0:
            return vertices, faces

        # Construct dual graph (faces as nodes, edges as edges)
        dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(
            face2edge
        )
        dual_edge2edge = edges[dual_edge2edge]
        dual_edges_weights = torch.norm(
            vertices[dual_edge2edge[:, 0]] - vertices[dual_edge2edge[:, 1]],
            dim=1,
        )

        # Mincut: Construct main graph and solve the mincut problem
        g = igraph.Graph()
        g.add_vertices(faces.shape[0])
        g.add_edges(dual_edges.cpu().numpy())
        g.es["weight"] = dual_edges_weights.cpu().numpy()

        g.add_vertex("s")  # source
        g.add_vertex("t")  # target

        g.add_edges(
            [(f, "s") for f in inner_face_indices],
            attributes={
                "weight": torch.ones(
                    inner_face_indices.shape[0], dtype=torch.float32
                )
                .cpu()
                .numpy()
            },
        )
        g.add_edges(
            [(f, "t") for f in outer_face_indices],
            attributes={
                "weight": torch.ones(
                    outer_face_indices.shape[0], dtype=torch.float32
                )
                .cpu()
                .numpy()
            },
        )

        cut = g.mincut("s", "t", (np.array(g.es["weight"]) * 1000).tolist())
        remove_face_indices = torch.tensor(
            [v for v in cut.partition[0] if v < faces.shape[0]],
            dtype=torch.long,
            device=faces.device,
        )

        # Check if the cut is valid with each connected component
        to_remove_cc = utils3d.torch.compute_connected_components(
            faces[remove_face_indices]
        )
        valid_remove_cc = []
        cutting_edges = []
        for cc in to_remove_cc:
            # Check visibility median for connected component
            visibility_median = visibility[remove_face_indices[cc]].median()
            if visibility_median > 0.25:
                continue

            # Check if the cutting loop is small enough
            cc_edge_indices, cc_edges_degree = torch.unique(
                face2edge[remove_face_indices[cc]], return_counts=True
            )
            cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
            cc_new_boundary_edge_indices = cc_boundary_edge_indices[
                ~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)
            ]
            if len(cc_new_boundary_edge_indices) > 0:
                cc_new_boundary_edge_cc = (
                    utils3d.torch.compute_edge_connected_components(
                        edges[cc_new_boundary_edge_indices]
                    )
                )
                cc_new_boundary_edges_cc_center = [
                    vertices[edges[cc_new_boundary_edge_indices[edge_cc]]]
                    .mean(dim=1)
                    .mean(dim=0)
                    for edge_cc in cc_new_boundary_edge_cc
                ]
                cc_new_boundary_edges_cc_area = []
                for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                    _e1 = (
                        vertices[
                            edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]
                        ]
                        - cc_new_boundary_edges_cc_center[i]
                    )
                    _e2 = (
                        vertices[
                            edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]
                        ]
                        - cc_new_boundary_edges_cc_center[i]
                    )
                    cc_new_boundary_edges_cc_area.append(
                        torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum()
                        * 0.5
                    )
                cutting_edges.append(cc_new_boundary_edge_indices)
                if any(
                    [
                        _l > max_hole_size
                        for _l in cc_new_boundary_edges_cc_area
                    ]
                ):
                    continue

            valid_remove_cc.append(cc)

        if len(valid_remove_cc) > 0:
            remove_face_indices = remove_face_indices[
                torch.cat(valid_remove_cc)
            ]
            mask = torch.ones(
                faces.shape[0], dtype=torch.bool, device=faces.device
            )
            mask[remove_face_indices] = 0
            faces = faces[mask]
            faces, vertices = utils3d.torch.remove_unreferenced_vertices(
                faces, vertices
            )

            tqdm.write(f"Removed {(~mask).sum()} faces by mincut")
        else:
            tqdm.write(f"Removed 0 faces by mincut")

        # Fill small boundaries (holes)
        mesh = _meshfix.PyTMesh()
        mesh.load_array(vertices.cpu().numpy(), faces.cpu().numpy())
        mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)

        _vertices, _faces = mesh.return_arrays()
        vertices = torch.tensor(_vertices).to(vertices)
        faces = torch.tensor(_faces).to(faces)

        return vertices, faces

    @property
    def vertices_np(self) -> np.ndarray:
        return self.vertices.cpu().numpy()

    @property
    def faces_np(self) -> np.ndarray:
        return self.faces.cpu().numpy()

    @log_mesh_changes
    def simplify(self, ratio: float) -> None:
        """Simplify the mesh using quadric edge collapse decimation.

        Args:
            ratio (float): Ratio of faces to filter out.
        """
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Simplify ratio must be between 0 and 1.")

        # Convert to PyVista format for simplification
        mesh = pv.PolyData(
            self.vertices_np,
            np.hstack([np.full((self.faces.shape[0], 1), 3), self.faces_np]),
        )
        mesh = mesh.decimate(ratio, progress_bar=True)

        # Update vertices and faces
        self.vertices = torch.tensor(
            mesh.points, device=self.device, dtype=torch.float32
        )
        self.faces = torch.tensor(
            mesh.faces.reshape(-1, 4)[:, 1:],
            device=self.device,
            dtype=torch.int32,
        )

    @spaces.GPU
    def __call__(
        self,
        filter_ratio: float,
        max_hole_size: float,
        resolution: int,
        num_views: int,
        norm_mesh_ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process the mesh by simplifying and filling holes.

        This method performs a two-step process:
        1. Simplifies mesh by reducing faces using quadric edge decimation.
        2. Fills holes by removing invisible faces, repairing small boundaries.

        Args:
            filter_ratio (float): Ratio of faces to simplify out.
                Must be in the range (0, 1).
            max_hole_size (float): Maximum area of a hole to fill. Connected
                components of holes larger than this size will not be repaired.
            resolution (int): Resolution of the rasterization buffer.
            num_views (int): Number of viewpoints to sample for rasterization.
            norm_mesh_ratio (float, optional): A scaling factor applied to the
                vertices of the mesh during processing.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - vertices: Simplified and repaired vertex array of (V, 3).
                - faces: Simplified and repaired face array of (F, 3).
        """
        self.vertices = self.vertices.to(self.device)
        self.faces = self.faces.to(self.device)

        self.simplify(ratio=filter_ratio)
        self.fill_holes(
            max_hole_size=max_hole_size,
            max_hole_nbe=int(250 * np.sqrt(1 - filter_ratio)),
            resolution=resolution,
            num_views=num_views,
            norm_mesh_ratio=norm_mesh_ratio,
        )

        return self.vertices_np, self.faces_np
