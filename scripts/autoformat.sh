#!/bin/bash

ROOT_DIR=${1}

set -e

black --config=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
isort  --settings-file=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
pycodestyle --show-source --config=${ROOT_DIR}setup.cfg ${ROOT_DIR}./
pydocstyle --config=${ROOT_DIR}pyproject.toml  ${ROOT_DIR}./
