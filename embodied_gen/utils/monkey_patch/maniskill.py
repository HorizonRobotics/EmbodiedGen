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


import numpy as np
import torch


def monkey_patch_maniskill():
    """Monkey patches ManiSkillScene to support sensor image retrieval and RGBA rendering."""
    from mani_skill.envs.scene import ManiSkillScene

    def get_sensor_images(
        self, obs: dict[str, any]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Retrieve images from all sensors based on observations."""
        sensor_data = dict()
        for name, sensor in self.sensors.items():
            sensor_data[name] = sensor.get_images(obs[name])
        return sensor_data

    def get_human_render_camera_images(
        self, camera_name: str = None, return_alpha: bool = False
    ) -> dict[str, torch.Tensor]:
        """Render images from human-view cameras, optionally generating alpha channel from segmentation."""

        def get_rgba_tensor(camera, return_alpha):
            color = camera.get_obs(
                rgb=True, depth=False, segmentation=False, position=False
            )["rgb"]
            if return_alpha:
                seg_labels = camera.get_obs(
                    rgb=False, depth=False, segmentation=True, position=False
                )["segmentation"]
                masks = np.where((seg_labels.cpu() > 1), 255, 0).astype(
                    np.uint8
                )
                masks = torch.tensor(masks).to(color.device)
                color = torch.concat([color, masks], dim=-1)

            return color

        image_data = dict()
        if self.gpu_sim_enabled:
            if self.parallel_in_single_scene:
                for name, camera in self.human_render_cameras.items():
                    camera.camera._render_cameras[0].take_picture()
                    rgba = get_rgba_tensor(camera, return_alpha)
                    image_data[name] = rgba
            else:
                for name, camera in self.human_render_cameras.items():
                    if camera_name is not None and name != camera_name:
                        continue
                    assert camera.config.shader_config.shader_pack not in [
                        "rt",
                        "rt-fast",
                        "rt-med",
                    ], "ray tracing shaders do not work with parallel rendering"
                    camera.capture()
                    rgba = get_rgba_tensor(camera, return_alpha)
                    image_data[name] = rgba
        else:
            for name, camera in self.human_render_cameras.items():
                if camera_name is not None and name != camera_name:
                    continue
                camera.capture()
                rgba = get_rgba_tensor(camera, return_alpha)
                image_data[name] = rgba

        return image_data

    ManiSkillScene.get_sensor_images = get_sensor_images
    ManiSkillScene.get_human_render_camera_images = (
        get_human_render_camera_images
    )
