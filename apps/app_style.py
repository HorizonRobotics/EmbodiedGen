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

from gradio.themes import Soft
from gradio.themes.utils.colors import gray, neutral, slate, stone, teal, zinc

lighting_css = """
<style>
#lighter_mesh canvas {
    filter: brightness(2.3) !important;
}
</style>
"""

image_css = """
<style>
.image_fit .image-frame {
object-fit: contain !important;
height: 100% !important;
}
</style>
"""

custom_theme = Soft(
    primary_hue=stone,
    secondary_hue=gray,
    radius_size="md",
    text_size="sm",
    spacing_size="sm",
)
