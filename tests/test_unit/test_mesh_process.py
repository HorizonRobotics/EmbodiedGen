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
import tempfile
from time import time

import pytest
from embodied_gen.data.convex_decomposer import decompose_convex_mesh

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "input_mesh_path, max_convex_hull",
    [
        ("apps/assets/example_texture/meshes/robot_text.obj", 8),
        # ("apps/assets/example_texture/meshes/robot_text.obj", 32),
        # ("apps/assets/example_texture/meshes/robot_text.obj", 64),
        # ("apps/assets/example_texture/meshes/robot_text.obj", 128),
    ],
)
def test_decompose_convex_mesh(input_mesh_path, max_convex_hull):
    d_params = dict(
        threshold=0.05, max_convex_hull=max_convex_hull, verbose=False
    )
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=True) as tmp_file:
        start_time = time()
        decompose_convex_mesh(input_mesh_path, tmp_file.name, **d_params)
        logger.info(f"Finished in {time()-start_time:.2f}s")
