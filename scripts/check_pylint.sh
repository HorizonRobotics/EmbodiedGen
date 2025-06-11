#!/bin/bash

ROOT_DIR=${1}

set -e


pycodestyle --show-source --config=${ROOT_DIR}setup.cfg ${ROOT_DIR}./
pydocstyle --config=${ROOT_DIR}pyproject.toml  ${ROOT_DIR}./
black --check --diff  --config=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
isort --diff   --settings-file=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
