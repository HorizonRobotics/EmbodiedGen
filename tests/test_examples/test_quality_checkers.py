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


import logging
import tempfile
from glob import glob

import pytest
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.process_media import (
    combine_images_to_grid,
    render_asset3d,
)
from embodied_gen.validators.quality_checkers import (
    ImageAestheticChecker,
    ImageSegChecker,
    MeshGeoChecker,
    PanoHeightEstimator,
    PanoImageGenChecker,
    PanoImageOccChecker,
    SemanticConsistChecker,
    TextGenAlignChecker,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def geo_checker():
    return MeshGeoChecker(GPT_CLIENT)


@pytest.fixture(scope="module")
def seg_checker():
    return ImageSegChecker(GPT_CLIENT)


@pytest.fixture(scope="module")
def aesthetic_checker():
    return ImageAestheticChecker()


@pytest.fixture(scope="module")
def semantic_checker():
    return SemanticConsistChecker(GPT_CLIENT)


@pytest.fixture(scope="module")
def textalign_checker():
    return TextGenAlignChecker(GPT_CLIENT)


@pytest.fixture(scope="module")
def pano_checker():
    return PanoImageGenChecker(GPT_CLIENT)


@pytest.fixture(scope="module")
def pano_height_estimator():
    return PanoHeightEstimator(GPT_CLIENT)


@pytest.fixture(scope="module")
def panoocc_checker():
    return PanoImageOccChecker(GPT_CLIENT, box_hw=[90, 1000])


def test_geo_checker(geo_checker):
    from embodied_gen.utils.process_media import combine_images_to_grid

    image_paths = glob(
        "outputs/layouts_gens3/task_0000/asset3d/pen/result/renders/image_color/*png"
    )
    image = combine_images_to_grid(image_paths)[0]
    # image = "outputs/utest_cases/geo_checker/color.png"
    flag, result = geo_checker(
        [
            image,
        ],
        text="pen",
    )
    logger.info(f"geo_checker: {flag}, {result}")
    assert isinstance(flag, bool)
    assert isinstance(result, str)


def test_aesthetic_checker(aesthetic_checker):
    flag, result = aesthetic_checker("apps/assets/example_image/sample_02.jpg")
    logger.info(f"aesthetic_checker: {flag}, {result}")
    assert isinstance(flag, bool)
    assert isinstance(result, float)


def test_seg_checker(seg_checker):
    flag, result = seg_checker(
        [
            "apps/assets/example_image/sample_02.jpg",
            "apps/assets/example_image/sample_02.jpg",
        ]
    )
    logger.info(f"seg_checker: {flag}, {result}")
    assert isinstance(flag, bool)
    assert isinstance(result, str)


def test_semantic_checker(semantic_checker):
    test_cases = [
        ("table", "outputs/utest_cases/semantic_checker/task_0001_table.png"),
        ("table", "outputs/utest_cases/semantic_checker/task_0002_table.png"),
        (
            "banana",
            "outputs/utest_cases/semantic_checker/task_0002_banana.png",
        ),
        (
            "napkin",
            "outputs/utest_cases/semantic_checker/task_0003_napkin.png",
        ),
        ("table", "outputs/utest_cases/semantic_checker/task_0003_table.png"),
        ("fork", "outputs/utest_cases/semantic_checker/task_0005_fork.png"),
        ("spoon", "outputs/utest_cases/semantic_checker/task_0005_spoon.png"),
        ("table", "outputs/utest_cases/semantic_checker/task_0007_table.png"),
        ("table", "outputs/utest_cases/semantic_checker/task_0008_table.png"),
        ("table", "outputs/utest_cases/semantic_checker/task_0009_table.png"),
        ("table", "outputs/utest_cases/semantic_checker/task_0011_table.png"),
        ("desk", "outputs/utest_cases/semantic_checker/task_0012_desk.png"),
        (
            "laptop",
            "outputs/utest_cases/semantic_checker/task_0012_laptop.png",
        ),
        ("table", "outputs/utest_cases/semantic_checker/task_0014_table.png"),
        ("desk", "outputs/utest_cases/semantic_checker/task_0016_desk.png"),
        ("shelf", "outputs/utest_cases/semantic_checker/task_0018_shelf.png"),
        ("table", "outputs/utest_cases/semantic_checker/task_0000_table.png"),
        ("pen", "outputs/layouts_gens2/task_0000/images/pen_raw.png"),
    ]
    for test_case in test_cases:
        flag, result = semantic_checker(
            text=test_case[0],
            image=[test_case[1]],
        )
        logger.info(f"semantic_checker: {flag}, {result}, {test_case[1]}")
        assert isinstance(flag, bool)
        assert isinstance(result, str)


@pytest.mark.parametrize(
    "mesh_path, text_desc",
    [
        ("apps/assets/example_texture/meshes/chair.obj", "chair"),
        ("apps/assets/example_texture/meshes/clock.obj", "clock"),
    ],
)
def test_textgen_checker(textalign_checker, mesh_path, text_desc):
    with tempfile.TemporaryDirectory() as output_root:
        image_list = render_asset3d(
            mesh_path,
            output_root=output_root,
            num_images=4,
            elevation=(30, -30),
            output_subdir="renders",
            no_index_file=True,
            with_mtl=False,
        )
        image_list = combine_images_to_grid(image_list)
        flag, result = textalign_checker(text_desc, image_list)
        logger.info(f"textalign_checker: {flag}, {result}")


def test_panoheight_estimator(pano_height_estimator):
    image_paths = glob("outputs/bg_scenes/*/*.png")
    for image_path in image_paths:
        result = pano_height_estimator(image_path)
        logger.info(f"{type(result)}, {result}")


def test_pano_checker(pano_checker):
    image_paths = glob("outputs/bg_scenes/*/*.png")
    for image_path in image_paths:
        flag, result = pano_checker(image_path)
        logger.info(f"{image_path} {flag}, {result}")


def test_panoocc_checker(panoocc_checker):
    image_paths = glob("outputs/bg_scenes/*/*.png")
    for image_path in image_paths:
        flag, result = panoocc_checker(image_path)
        logger.info(f"{image_path} {flag}, {result}")
