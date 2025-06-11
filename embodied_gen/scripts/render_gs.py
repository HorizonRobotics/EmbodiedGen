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


import argparse
import logging
import math
import os

import cv2
import numpy as np
import spaces
import torch
from tqdm import tqdm
from embodied_gen.data.utils import (
    CameraSetting,
    init_kal_camera,
    normalize_vertices_array,
)
from embodied_gen.models.gs_model import GaussianOperator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Render GS color images")

    parser.add_argument(
        "--input_gs", type=str, help="Input render GS.ply path."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output grid image path for rendered GS color images.",
    )
    parser.add_argument(
        "--num_images", type=int, default=6, help="Number of images to render."
    )
    parser.add_argument(
        "--elevation",
        type=float,
        nargs="+",
        default=[20.0, -10.0],
        help="Elevation angles for the camera (default: [20.0, -10.0])",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=5,
        help="Camera distance (default: 5)",
    )
    parser.add_argument(
        "--resolution_hw",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Resolution of the output images (default: (512, 512))",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=30,
        help="Field of view in degrees (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run on (default: `cuda`)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Output image size for single view in color grid (default: 512)",
    )

    args, unknown = parser.parse_known_args()

    return args


def load_gs_model(
    input_gs: str, pre_quat: list[float] = [0.0, 0.7071, 0.0, -0.7071]
) -> GaussianOperator:
    gs_model = GaussianOperator.load_from_ply(input_gs)
    # Normalize vertices to [-1, 1], center to (0, 0, 0).
    _, scale, center = normalize_vertices_array(gs_model._means)
    scale, center = float(scale), center.tolist()
    transpose = [*[-v for v in center], *pre_quat]
    instance_pose = torch.tensor(transpose).to(gs_model.device)
    gs_model = gs_model.get_gaussians(instance_pose=instance_pose)
    gs_model.rescale(scale)

    return gs_model


@spaces.GPU
def entrypoint(input_gs: str = None, output_path: str = None) -> None:
    args = parse_args()
    if isinstance(input_gs, str):
        args.input_gs = input_gs
    if isinstance(output_path, str):
        args.output_path = output_path

    # Setup camera parameters
    camera_params = CameraSetting(
        num_images=args.num_images,
        elevation=args.elevation,
        distance=args.distance,
        resolution_hw=args.resolution_hw,
        fov=math.radians(args.fov),
        device=args.device,
    )
    camera = init_kal_camera(camera_params)
    matrix_mv = camera.view_matrix()  # (n_cam 4 4) world2cam
    matrix_mv[:, :3, 3] = -matrix_mv[:, :3, 3]
    w2cs = matrix_mv.to(camera_params.device)
    c2ws = [torch.linalg.inv(matrix) for matrix in w2cs]
    Ks = torch.tensor(camera_params.Ks).to(camera_params.device)

    # Load GS model and normalize.
    gs_model = load_gs_model(args.input_gs, pre_quat=[0.0, 0.0, 1.0, 0.0])

    # Render GS color images.
    images = []
    for idx in tqdm(range(len(c2ws)), desc="Rendering GS"):
        result = gs_model.render(
            c2ws[idx],
            Ks=Ks,
            image_width=camera_params.resolution_hw[1],
            image_height=camera_params.resolution_hw[0],
        )
        color = cv2.resize(
            result.rgba,
            (args.image_size, args.image_size),
            interpolation=cv2.INTER_AREA,
        )
        images.append(color)

    # Cat color images into grid image and save.
    select_idxs = [[0, 2, 1], [5, 4, 3]]  # fix order for 6 views
    grid_image = []
    for row_idxs in select_idxs:
        row_image = []
        for row_idx in row_idxs:
            row_image.append(images[row_idx])
        row_image = np.concatenate(row_image, axis=1)
        grid_image.append(row_image)

    grid_image = np.concatenate(grid_image, axis=0)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, grid_image)
    logger.info(f"Saved grid image to {args.output_path}")


if __name__ == "__main__":
    entrypoint()
