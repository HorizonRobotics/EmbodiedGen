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
import signal
import subprocess
import sys
from pathlib import Path

from embodied_gen.utils.log import logger


def get_blender_python_path():
    """Resolve path to Blender's bundled Python binary from env or default location."""
    env_path = os.environ.get("BLENDER_PYTHON_BIN")
    if env_path and os.path.exists(env_path):
        return env_path

    default_relative_path = Path(
        "thirdparty/infinigen/blender/4.2/python/bin/python3.11"
    )
    if default_relative_path.exists():
        return str(default_relative_path.resolve())

    logger.error("Error: Could not find Blender Python binary.")
    logger.error("Please set the BLENDER_PYTHON_BIN environment variable.")
    sys.exit(1)


def entrypoint():
    """Entry point wrapper to execute script within Blender's Python environment."""
    blender_python = get_blender_python_path()
    args = sys.argv[1:]
    process = subprocess.Popen([blender_python] + args, start_new_session=True)
    try:
        return_code = process.wait()
        sys.exit(return_code)

    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        logger.error("\nProgram interrupted by user (Cmd+C). Exiting.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    entrypoint()
