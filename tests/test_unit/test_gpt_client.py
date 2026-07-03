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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from PIL import Image
from embodied_gen.utils.gpt_clients import CONFIG_FILE, GPTclient


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def env_vars(monkeypatch, config):
    agent_type = config["agent_type"]
    agent_config = config.get(agent_type, {})
    monkeypatch.setenv(
        "ENDPOINT", agent_config.get("endpoint", "fake_endpoint")
    )
    monkeypatch.setenv("API_KEY", agent_config.get("api_key", "fake_api_key"))
    monkeypatch.setenv("API_VERSION", agent_config.get("api_version", "v1"))
    monkeypatch.setenv(
        "MODEL_NAME", agent_config.get("model_name", "test_model")
    )
    yield


@pytest.fixture
def gpt_client(env_vars):
    client = GPTclient(
        endpoint=os.environ.get("ENDPOINT"),
        api_key=os.environ.get("API_KEY"),
        api_version=os.environ.get("API_VERSION"),
        model_name=os.environ.get("MODEL_NAME"),
        check_connection=False,
    )
    return client


@pytest.mark.parametrize(
    "text_prompt, image_base64",
    [
        ("What is the capital of China?", None),
        (
            "What is the content in each image?",
            "apps/assets/example_image/sample_02.jpg",
        ),
        (
            "What is the content in each image?",
            [
                "apps/assets/example_image/sample_02.jpg",
                "apps/assets/example_image/sample_03.jpg",
            ],
        ),
        (
            "What is the content in each image?",
            [
                Image.new("RGB", (64, 64), "red"),
                Image.new("RGB", (64, 64), "blue"),
            ],
        ),
    ],
)
def test_gptclient_query(gpt_client, text_prompt, image_base64):
    # mock GPTclient.query
    with patch.object(
        GPTclient, "query", return_value="mocked response"
    ) as mock_query:
        response = gpt_client.query(
            text_prompt=text_prompt, image_base64=image_base64
        )
        assert response == "mocked response"
        mock_query.assert_called_once_with(
            text_prompt=text_prompt, image_base64=image_base64
        )


def _make_client(model_name: str) -> GPTclient:
    return GPTclient(
        endpoint="https://yfb-openai-sweden.openai.azure.com/",
        api_key="fake_key",
        api_version="2024-12-01-preview",
        model_name=model_name,
        check_connection=False,
    )


def _fake_response(text: str = "ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_is_gpt5_model_detection():
    assert GPTclient._is_gpt5_model("gpt-5.4") is True
    assert GPTclient._is_gpt5_model("GPT-5") is True
    assert GPTclient._is_gpt5_model("gpt5-turbo") is True
    assert GPTclient._is_gpt5_model("gpt-4o") is False
    assert (
        GPTclient._is_gpt5_model("qwen/qwen2.5-vl-72b-instruct:free") is False
    )


def test_gpt5_text_payload_uses_max_completion_tokens():
    client = _make_client("gpt-5.4")
    with patch.object(
        client, "completion_with_backoff", return_value=_fake_response("hi")
    ) as mock_call:
        out = client.query("Hello world")
    assert out == "hi"
    payload = mock_call.call_args.kwargs
    assert payload["model"] == "gpt-5.4"
    assert payload["max_completion_tokens"] == 8192
    # GPT-5 path should NOT include legacy sampling params.
    for forbidden in (
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
    ):
        assert forbidden not in payload, (
            f"{forbidden} leaked into gpt-5 payload"
        )
    # Text-only: a single user content block of type text.
    user_content = payload["messages"][-1]["content"]
    assert user_content[0]["type"] == "text"
    assert all(c["type"] != "image_url" for c in user_content)


def test_gpt5_text_image_payload():
    client = _make_client("gpt-5.4")
    img1 = Image.new("RGB", (32, 32), "red")
    img2 = Image.new("RGB", (32, 32), "blue")
    with patch.object(
        client, "completion_with_backoff", return_value=_fake_response("ok")
    ) as mock_call:
        client.query("describe", image_base64=[img1, img2])
    payload = mock_call.call_args.kwargs
    user_content = payload["messages"][-1]["content"]
    image_blocks = [c for c in user_content if c["type"] == "image_url"]
    assert len(image_blocks) == 2
    for b in image_blocks:
        assert b["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["max_completion_tokens"] == 8192


def test_non_gpt5_payload_keeps_legacy_params():
    client = _make_client("yfb-gpt-4o")
    with patch.object(
        client, "completion_with_backoff", return_value=_fake_response("ok")
    ) as mock_call:
        client.query("Hello")
    payload = mock_call.call_args.kwargs
    assert payload["max_tokens"] == 500
    assert payload["temperature"] == 0.1
    assert "max_completion_tokens" not in payload


def test_params_override():
    client = _make_client("gpt-5.4")
    with patch.object(
        client, "completion_with_backoff", return_value=_fake_response("ok")
    ) as mock_call:
        client.query("Hello", params={"max_completion_tokens": 256})
    payload = mock_call.call_args.kwargs
    assert payload["max_completion_tokens"] == 256


def test_gpt5_filters_unsupported_params():
    client = _make_client("gpt-5.4")
    with patch.object(
        client, "completion_with_backoff", return_value=_fake_response("ok")
    ) as mock_call:
        client.query(
            "Hello",
            params={
                "temperature": 1.0,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "max_tokens": 256,
            },
        )
    payload = mock_call.call_args.kwargs
    for forbidden in (
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "max_tokens",
    ):
        assert forbidden not in payload
    # max_tokens should be migrated to max_completion_tokens.
    assert payload["max_completion_tokens"] == 256


def test_image_path_loaded_from_disk(tmp_path):
    img_path = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), "green").save(img_path)
    client = _make_client("gpt-5.4")
    with patch.object(
        client, "completion_with_backoff", return_value=_fake_response("ok")
    ) as mock_call:
        client.query("describe", image_base64=str(img_path))
    user_content = mock_call.call_args.kwargs["messages"][-1]["content"]
    image_blocks = [c for c in user_content if c["type"] == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
