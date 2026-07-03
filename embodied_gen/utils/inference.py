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


from embodied_gen.utils.monkey_patch.trellis import monkey_path_trellis

monkey_path_trellis()
import random

import torch
from PIL import Image
from embodied_gen.data.utils import trellis_preprocess
from embodied_gen.models.sam3d import Sam3dInference
from embodied_gen.utils.trender import pack_state, unpack_state
from thirdparty.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline

__all__ = [
    "image3d_model_infer",
]


def image3d_model_infer(
    pipe: TrellisImageTo3DPipeline | Sam3dInference,
    seg_image: Image.Image,
    seed: int = None,
    **kwargs: dict,
) -> dict[str, any]:
    """Execute 3D generation using Trellis or SAM3D pipeline on input image."""
    if isinstance(pipe, TrellisImageTo3DPipeline):
        pipe.cuda()
        seg_image = trellis_preprocess(seg_image)
        outputs = pipe.run(
            seg_image,
            preprocess_image=False,
            seed=(random.randint(0, 100000) if seed is None else seed),
            # Optional parameters
            # sparse_structure_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 3,
            # },
            **kwargs,
        )
        pipe.cpu()
    elif isinstance(pipe, Sam3dInference):
        outputs = pipe.run(
            seg_image,
            seed=(random.randint(0, 100000) if seed is None else seed),
            # stage1_inference_steps=25,
            # stage2_inference_steps=25,
            **kwargs,
        )
        state = pack_state(outputs["gaussian"][0], outputs["mesh"][0])
        # Align GS3D from SAM3D with TRELLIS format.
        outputs["gaussian"][0], _ = unpack_state(state, device="cuda")
    else:
        raise ValueError(f"Unsupported pipeline type: {type(pipe)}")

    torch.cuda.empty_cache()

    return outputs
