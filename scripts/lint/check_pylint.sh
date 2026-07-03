#!/bin/bash

ROOT_DIR=${1}

set -e


ruff check --config=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
ruff format --check --diff --config=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
