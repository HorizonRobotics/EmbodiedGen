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


# Wrapper to run infinigen_examples.generate_indoors with correct gin registration.
# Importing (rather than python -m) ensures the module loads with full module path,
# avoiding the "Ambiguous selector 'compose_indoors'" error from __main__.
from embodied_gen.utils.monkey_patch.infinigen import (
    add_run_main_to_module,
    monkey_patch_infinigen,
)

monkey_patch_infinigen()

import infinigen_examples.generate_indoors as gi

add_run_main_to_module(gi)
gi._run_main()
