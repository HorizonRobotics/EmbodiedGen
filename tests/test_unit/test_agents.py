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


from unittest.mock import patch

import pytest
from PIL import Image
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.validators.quality_checkers import (
    ImageSegChecker,
    MeshGeoChecker,
    PanoHeightEstimator,
    PanoImageGenChecker,
    SemanticConsistChecker,
)


@pytest.fixture(autouse=True)
def gptclient_query():
    with patch.object(
        GPT_CLIENT, "query", return_value="mocked gpt response"
    ) as mock:
        yield mock


@pytest.fixture()
def gptclient_query_case2():
    with patch.object(GPT_CLIENT, "query", return_value=None) as mock:
        yield mock


@pytest.mark.parametrize(
    "input_images",
    [
        "dummy_path/color_grid_6view.png",
        ["dummy_path/color_grid_6view.jpg"],
        [
            "dummy_path/color_grid_6view.png",
            "dummy_path/color_grid_6view2.png",
        ],
        [
            Image.new("RGB", (64, 64), "red"),
            Image.new("RGB", (64, 64), "blue"),
        ],
    ],
)
def test_geo_checker_varied_inputs(input_images):
    geo_checker = MeshGeoChecker(GPT_CLIENT)
    flag, result = geo_checker(input_images)
    assert isinstance(flag, (bool, type(None)))
    assert isinstance(result, str)


def test_seg_checker():
    seg_checker = ImageSegChecker(GPT_CLIENT)
    flag, result = seg_checker(
        [
            "dummy_path/sample_0_raw.png",  # raw image
            "dummy_path/sample_0_cond.png",  # segmented image
        ]
    )
    assert isinstance(flag, (bool, type(None)))
    assert isinstance(result, str)


def test_semantic_checker():
    semantic_checker = SemanticConsistChecker(GPT_CLIENT)
    flag, result = semantic_checker(
        text="pen",
        image=["dummy_path/pen.png"],
    )
    assert isinstance(flag, (bool, type(None)))
    assert isinstance(result, str)


def test_semantic_checker(gptclient_query_case2):
    semantic_checker = SemanticConsistChecker(GPT_CLIENT)
    flag, result = semantic_checker(
        text="pen",
        image=["dummy_path/pen.png"],
    )
    assert isinstance(flag, (bool, type(None)))
    assert isinstance(result, str)


def test_panoheight_estimator():
    checker = PanoHeightEstimator(GPT_CLIENT, default_value=3.5)
    result = checker(image_paths="dummy_path/pano.png")
    assert isinstance(result, float)


def test_panogen_checker():
    checker = PanoImageGenChecker(GPT_CLIENT)
    flag, result = checker(image_paths="dummy_path/pano.png")
    assert isinstance(flag, (bool, type(None)))
    assert isinstance(result, str)
