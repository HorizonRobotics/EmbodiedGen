#!/bin/bash

ROOT_DIR=${1}

set -e

ruff check --fix --config=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
ruff format --config=${ROOT_DIR}pyproject.toml ${ROOT_DIR}./
