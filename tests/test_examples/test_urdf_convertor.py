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
        # "outputs/layouts_test/task_0004/asset3d/pen/result/mesh/pen.obj",
        # "outputs/layouts_test/task_0004/asset3d/notepad/result/mesh/notepad.obj",
        # "outputs/layouts_test/task_0005/asset3d/plate/result/mesh/plate.obj",
        # "outputs/layouts_test/task_0005/asset3d/spoon/result/mesh/spoon.obj",
        # "outputs/layouts_test/task_0007/asset3d/notebook/result/mesh/notebook.obj",
        # "outputs/layouts_test/task_0008/asset3d/plate/result/mesh/plate.obj",
        # "outputs/layouts_test/task_0008/asset3d/spoon/result/mesh/spoon.obj",
        # "outputs/layouts_test/task_0009/asset3d/book/result/mesh/book.obj",
        # "outputs/layouts_test/task_0009/asset3d/lamp/result/mesh/lamp.obj",
        # "outputs/layouts_test/task_0009/asset3d/remote_control/result/mesh/remote_control.obj",
        # "outputs/layouts_test/task_0012/asset3d/keyboard/result/mesh/keyboard.obj",
        # "outputs/layouts_test/task_0012/asset3d/mouse/result/mesh/mouse.obj",
        # "outputs/layouts_test/task_0013/asset3d/table/result/mesh/table.obj",
        # "outputs/layouts_test/task_0015/asset3d/marker/result/mesh/marker.obj",
        "outputs/layouts_test2/task_0000/asset3d/notepad/result/mesh/notepad.obj",
    ]
    for idx, mesh_path in enumerate(mesh_paths):
        filename = mesh_path.split("/")[-1].split(".")[0]
        urdf_path = urdf_gen(
            mesh_path=mesh_path,
            output_root=f"outputs/test_urdf/sample_{idx}",
            category="notepad",
            # min_height=1.0,
            # max_height=1.2,
        )
