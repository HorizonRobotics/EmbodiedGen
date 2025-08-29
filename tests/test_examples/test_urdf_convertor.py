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

from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.validators.urdf_convertor import URDFGenerator


def test_urdf_convertor():
    urdf_gen = URDFGenerator(GPT_CLIENT, render_view_num=4)
    mesh_paths = [
        "outputs/test_urdf/sample_0/mesh/pen.obj",
        "outputs/test_urdf/sample_1/mesh/notepad.obj",
        "outputs/test_urdf/sample_2/mesh/plate.obj",
        "outputs/test_urdf/sample_3/mesh/spoon.obj",
        "outputs/test_urdf/sample_4/mesh/notebook.obj",
        "outputs/test_urdf/sample_5/mesh/plate.obj",
        "outputs/test_urdf/sample_6/mesh/spoon.obj",
        "outputs/test_urdf/sample_7/mesh/book.obj",
        "outputs/test_urdf/sample_8/mesh/lamp.obj",
        "outputs/test_urdf/sample_9/mesh/remote_control.obj",
        "outputs/test_urdf/sample_10/mesh/keyboard.obj",
        "outputs/test_urdf/sample_11/mesh/mouse.obj",
        "outputs/test_urdf/sample_12/mesh/table.obj",
        "outputs/test_urdf/sample_13/mesh/marker.obj",
        "outputs/test_urdf/pen/result/mesh/pen.obj",
        "outputs/test_urdf/notebook/result/mesh/notebook.obj",
        "outputs/test_urdf/marker/result/mesh/marker.obj",
        "outputs/test_urdf/pen2/result/mesh/pen.obj",
    ]
    for idx, mesh_path in enumerate(mesh_paths):
        filename = mesh_path.split("/")[-1].split(".")[0]
        urdf_path = urdf_gen(
            mesh_path=mesh_path,
            output_root=f"outputs/test_urdf2/sample_{idx}",
            category=filename,
            # min_height=1.0,
            # max_height=1.2,
        )


def test_decompose_convex_mesh():
    urdf_gen = URDFGenerator(GPT_CLIENT, decompose_convex=True)
    mesh_paths = [
        "outputs/test_urdf/sample_0/mesh/pen.obj",
        "outputs/test_urdf/sample_1/mesh/notepad.obj",
        "outputs/test_urdf/sample_2/mesh/plate.obj",
    ]
    for idx, mesh_path in enumerate(mesh_paths):
        filename = mesh_path.split("/")[-1].split(".")[0]
        urdf_path = urdf_gen(
            mesh_path=mesh_path,
            output_root=f"outputs/test_urdf2/sample_{idx}",
            category=filename,
        )
