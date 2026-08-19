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

import torch

_BLACKWELL_MINIMUM_COMPUTE_CAPABILITY = (12, 0)


def disable_xformers_flash3() -> bool:
    """Remove FlashAttention 3 from the xFormers dispatcher."""
    try:
        from xformers.ops.fmha import _set_use_fa3
    except (ImportError, AttributeError):
        return False

    _set_use_fa3(False)
    return True


def disable_xformers_flash3_on_blackwell() -> bool:
    """Disable xFormers FlashAttention 3 when a Blackwell GPU is visible."""
    if not torch.cuda.is_available():
        return False

    for device_index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(device_index)
        if capability >= _BLACKWELL_MINIMUM_COMPUTE_CAPABILITY:
            return disable_xformers_flash3()

    return False
