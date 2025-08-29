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
import multiprocessing as mp
import os

import coacd
import trimesh

logger = logging.getLogger(__name__)

__all__ = [
    "decompose_convex_coacd",
    "decompose_convex_mesh",
    "decompose_convex_process",
]


def decompose_convex_coacd(
    filename: str, outfile: str, params: dict, verbose: bool = False
) -> None:
    coacd.set_log_level("info" if verbose else "warn")

    mesh = trimesh.load(filename, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    result = coacd.run_coacd(mesh, **params)
    combined = sum([trimesh.Trimesh(*m) for m in result])
    combined.export(outfile)


def decompose_convex_mesh(
    filename: str,
    outfile: str,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 30,
    resolution: int = 2000,
    mcts_nodes: int = 20,
    mcts_iterations: int = 150,
    mcts_max_depth: int = 3,
    pca: bool = False,
    merge: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> str:
    """Decompose a mesh into convex parts using the CoACD algorithm."""
    coacd.set_log_level("info" if verbose else "warn")

    if os.path.exists(outfile):
        logger.warning(f"Output file {outfile} already exists, removing it.")
        os.remove(outfile)

    params = dict(
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=merge,
        seed=seed,
    )

    try:
        decompose_convex_coacd(filename, outfile, params, verbose)
        if os.path.exists(outfile):
            return outfile
    except Exception as e:
        if verbose:
            print(f"Decompose convex first attempt failed: {e}.")

    if preprocess_mode != "on":
        try:
            params["preprocess_mode"] = "on"
            decompose_convex_coacd(filename, outfile, params, verbose)
            if os.path.exists(outfile):
                return outfile
        except Exception as e:
            if verbose:
                print(
                    f"Decompose convex second attempt with preprocess_mode='on' failed: {e}"
                )

    raise RuntimeError(f"Convex decomposition failed on {filename}")


def decompose_convex_mp(
    filename: str,
    outfile: str,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 30,
    resolution: int = 2000,
    mcts_nodes: int = 20,
    mcts_iterations: int = 150,
    mcts_max_depth: int = 3,
    pca: bool = False,
    merge: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> str:
    """Decompose a mesh into convex parts using the CoACD algorithm in a separate process.

    See https://simulately.wiki/docs/toolkits/ConvexDecomp for details.
    """
    params = dict(
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=merge,
        seed=seed,
    )

    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=decompose_convex_coacd,
        args=(filename, outfile, params, verbose),
    )
    p.start()
    p.join()
    if p.exitcode == 0 and os.path.exists(outfile):
        return outfile

    if preprocess_mode != "on":
        params["preprocess_mode"] = "on"
        p = ctx.Process(
            target=decompose_convex_coacd,
            args=(filename, outfile, params, verbose),
        )
        p.start()
        p.join()
        if p.exitcode == 0 and os.path.exists(outfile):
            return outfile

    raise RuntimeError(f"Convex decomposition failed on {filename}")
