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

import os
import sys

import torch
import torch.nn.functional as F


def monkey_path_trellis():
    """Monkey patches TRELLIS with specific environment settings and Gaussian setup functions."""
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    sys.path.append(os.path.join(current_dir, "../../.."))

    from thirdparty.TRELLIS.trellis.representations import Gaussian
    from thirdparty.TRELLIS.trellis.representations.gaussian.general_utils import (
        build_scaling_rotation,
        inverse_sigmoid,
        strip_symmetric,
    )

    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.expanduser(
        "~/.cache/torch_extensions"
    )
    os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto'
    os.environ['ATTN_BACKEND'] = (
        "xformers"  # Can be 'flash-attn' or 'xformers'
    )
    from thirdparty.TRELLIS.trellis.modules.sparse import set_attn

    set_attn("xformers")

    def patched_setup_functions(self):
        """Configure activation functions and biases for Gaussian representation."""

        def inverse_softplus(x):
            return x + torch.log(-torch.expm1(-x))

        def build_covariance_from_scaling_rotation(
            scaling, scaling_modifier, rotation
        ):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = F.softplus
            self.inverse_scaling_activation = inverse_softplus

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = F.normalize

        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias)
        ).to(self.device)
        self.rots_bias = torch.zeros((4)).to(self.device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias)
        ).to(self.device)

    Gaussian.setup_functions = patched_setup_functions
