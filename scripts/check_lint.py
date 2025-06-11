import argparse
import os
import subprocess
import sys


def get_root():
    current_file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(current_file_path)
    for _ in range(2):
        root_path = os.path.dirname(root_path)
    return root_path


def cpp_lint(root_path: str):
    # run external python file to lint cpp
    subprocess.check_call(
        " ".join(
            [
                "python3",
                f"{root_path}/scripts/lint_src/lint.py",
                "--project=asset_recons",
                "--path",
                f"{root_path}/src/",
                f"{root_path}/include/",
                f"{root_path}/module/",
                "--exclude_path",
                f"{root_path}/module/web_viz/front_end/",
            ]
        ),
        shell=True,
    )


def python_lint(root_path: str, auto_format: bool = False):
    # run external python file to lint python
    subprocess.check_call(
        " ".join(
            [
                "bash",
                (
                    f"{root_path}/scripts/lint/check_pylint.sh"
                    if not auto_format
                    else f"{root_path}/scripts/lint/autoformat.sh"
                ),
                f"{root_path}/",
            ]
        ),
        shell=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check format.")
    parser.add_argument(
        "--auto_format", action="store_true", help="auto format python"
    )
    parser = parser.parse_args()
    root_path = get_root()
    cpp_lint(root_path)
    python_lint(root_path, parser.auto_format)
