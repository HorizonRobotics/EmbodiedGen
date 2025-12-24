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

import cv2
import spaces
import torch
from PIL import Image
from tqdm import tqdm
from embodied_gen.data.utils import (
    CameraSetting,
    init_kal_camera,
)
from embodied_gen.models.gs_model import load_gs_model
from embodied_gen.utils.process_media import combine_images_to_grid

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


@spaces.GPU
def entrypoint(**kwargs) -> None:
    args = parse_args()
    for k, v in kwargs.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    # Setup camera parameters
    camera_params = CameraSetting(
        num_images=args.num_images,
        elevation=args.elevation,
        distance=args.distance,
        resolution_hw=args.resolution_hw,
        fov=math.radians(args.fov),
        device=args.device,
    )
    camera = init_kal_camera(camera_params, flip_az=True)
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
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGBA)
        images.append(Image.fromarray(color))

    combine_images_to_grid(images, image_mode="RGBA")[0].save(args.output_path)

    logger.info(f"Saved grid image to {args.output_path}")


if __name__ == "__main__":
    entrypoint()
