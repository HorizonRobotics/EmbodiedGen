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


import base64
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import openai
import yaml
from openai import AzureOpenAI, OpenAI  # pip install openai
from PIL import Image
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


__all__ = [
    "GPTclient",
]

CONFIG_FILE = str(Path(__file__).with_name("gpt_config.yaml"))
DEFAULT_GPT_TIMEOUT = float(os.environ.get("GPT_TIMEOUT", 90))
# GPT-5.x counts reasoning tokens against this cap, so it must be high
# enough to leave room for both reasoning and the visible reply.
GPT5_DEFAULT_MAX_COMPLETION_TOKENS = 8192
_CODEX_DEFAULT_REASONING_EFFORT = "medium"
_CODEX_ENV_KEYS = {
    "ALL_PROXY",
    "CODEX_HOME",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "PATH",
}


def _resolve_agent_settings(
    config: dict, environ: Optional[dict[str, str]] = None
) -> dict:
    """Resolve one provider configuration with environment overrides."""
    environ = os.environ if environ is None else environ
    agent_type = config["agent_type"]
    agent_config = config.get(agent_type, {})
    provider_override = environ.get("GPT_PROVIDER")

    if provider_override is not None:
        agent_config = {}

    return {
        "endpoint": environ.get("ENDPOINT", agent_config.get("endpoint")),
        "api_key": environ.get("API_KEY", agent_config.get("api_key")),
        "api_version": environ.get(
            "API_VERSION", agent_config.get("api_version")
        ),
        "model_name": environ.get(
            "MODEL_NAME", agent_config.get("model_name")
        ),
        "provider": provider_override or agent_config.get("provider"),
    }


def _codex_subprocess_environment() -> dict[str, str]:
    """Return the minimal host environment required by Codex CLI."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _CODEX_ENV_KEYS
    }


def combine_images_to_grid(
    images: list[str | Image.Image],
    cat_row_col: tuple[int, int] = None,
    target_wh: tuple[int, int] = (512, 512),
    image_mode: str = "RGB",
) -> list[Image.Image]:
    n_images = len(images)
    if n_images == 1:
        return images

    if cat_row_col is None:
        n_col = math.ceil(math.sqrt(n_images))
        n_row = math.ceil(n_images / n_col)
    else:
        n_row, n_col = cat_row_col

    images = [
        Image.open(p).convert(image_mode) if isinstance(p, str) else p
        for p in images
    ]
    images = [img.resize(target_wh) for img in images]

    grid_w, grid_h = n_col * target_wh[0], n_row * target_wh[1]
    grid = Image.new(image_mode, (grid_w, grid_h), (0, 0, 0))

    for idx, img in enumerate(images):
        row, col = divmod(idx, n_col)
        grid.paste(img, (col * target_wh[0], row * target_wh[1]))

    return [grid]


class GPTclient:
    """A client to interact with GPT models via API or Codex CLI.

    Supports text and image prompts, connection checking, and configurable parameters.

    Args:
        endpoint (str): API endpoint URL.
        api_key (str): API key for authentication.
        model_name (str, optional): Model name to use.
        api_version (str, optional): API version (for Azure).
        check_connection (bool, optional): Whether to check API connection.
        verbose (bool, optional): Enable verbose logging.
        timeout (float, optional): Max seconds for a single GPT request.
        provider (str, optional): Backend provider. Use ``codex`` to reuse a
            local Codex CLI login; otherwise the existing Azure/OpenAI-
            compatible API selection is preserved.

    Example:
        ```sh
        export ENDPOINT="https://yfb-openai-sweden.openai.azure.com"
        export API_KEY="xxxxxx"
        export API_VERSION="2025-03-01-preview"
        export MODEL_NAME="yfb-gpt-4o-sweden"
        ```
        ```py
        from embodied_gen.utils.gpt_clients import GPT_CLIENT

        response = GPT_CLIENT.query("Describe the physics of a falling apple.")
        response = GPT_CLIENT.query(
            text_prompt="Describe the content in each image."
            image_base64=["path/to/image1.png", "path/to/image2.jpg"],
        )
        ```
    """

    def __init__(
        self,
        endpoint: Optional[str],
        api_key: Optional[str],
        model_name: Optional[str] = "yfb-gpt-4o",
        api_version: Optional[str] = None,
        check_connection: bool = True,
        verbose: bool = False,
        timeout: float = DEFAULT_GPT_TIMEOUT,
        provider: Optional[str] = None,
    ):
        self.provider = (
            provider or ("azure" if api_version else "openai")
        ).lower()
        self.codex_executable = None
        if self.provider == "codex":
            self.codex_executable = shutil.which("codex")
            if self.codex_executable is None:
                raise RuntimeError(
                    "Codex CLI was not found. Install Codex and run "
                    "`codex login` before using the Codex provider."
                )
            self.client = None
        elif self.provider == "azure" or api_version is not None:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=0,
            )
        else:
            self.client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            )

        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.image_formats = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        self.verbose = verbose
        if check_connection:
            self.check_connection()

        logger.info(f"Using GPT model: {self.model_name}.")

    def _materialize_codex_image(
        self,
        image: str | Image.Image,
        target_stem: Path,
    ) -> Path:
        """Normalize one Codex image input to a temporary PNG file."""
        target = target_stem.with_suffix(".png")
        if isinstance(image, Image.Image):
            image.convert("RGB").save(target, format="PNG")
            return target

        if not isinstance(image, str):
            raise TypeError(
                "Codex image input must be a path, base64 string, or PIL Image"
            )

        if not image.startswith("data:"):
            source = Path(image).expanduser()
            try:
                source_is_file = source.is_file()
            except OSError:
                source_is_file = False
            if source_is_file:
                try:
                    with Image.open(source) as source_image:
                        source_image.convert("RGB").save(target, format="PNG")
                except OSError as exc:
                    raise ValueError(f"Invalid image file: {image}") from exc
                return target
            if source.suffix.lower() in self.image_formats:
                raise FileNotFoundError(f"Image file not found: {image}")
            encoded = image
        else:
            header, separator, encoded = image.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ValueError("Image data URI must contain base64 data")

        try:
            image_data = base64.b64decode(encoded, validate=True)
            with Image.open(BytesIO(image_data)) as decoded_image:
                decoded_image.convert("RGB").save(target, format="PNG")
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Codex image input is neither an existing image nor valid base64"
            ) from exc
        return target

    def _query_codex(
        self,
        text_prompt: str,
        image_base64: Optional[str | Image.Image | list[str | Image.Image]],
        system_role: str,
        params: Optional[dict],
    ) -> str:
        """Run one non-interactive Codex CLI request."""
        params = params or {}
        with tempfile.TemporaryDirectory(prefix="embodiedgen-codex-") as tmp:
            tmp_path = Path(tmp)
            output_path = tmp_path / "response.txt"
            image_paths = []

            images = image_base64 or []
            if not isinstance(images, list):
                images = [images]
            for index, image in enumerate(images):
                image_paths.append(
                    self._materialize_codex_image(
                        image,
                        tmp_path / f"image_{index}",
                    )
                )

            prompt = f"{system_role}\n\nUser request:\n{text_prompt}"
            command = [
                self.codex_executable,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--output-last-message",
                str(output_path),
                "--cd",
                str(tmp_path),
            ]
            model_name = params.get("model", self.model_name)
            if model_name:
                command.extend(["--model", model_name])
            reasoning_effort = params.get(
                "model_reasoning_effort", _CODEX_DEFAULT_REASONING_EFFORT
            )
            if not isinstance(reasoning_effort, str) or not reasoning_effort:
                raise ValueError(
                    "model_reasoning_effort must be a non-empty string"
                )
            command.extend(
                [
                    "--config",
                    f"model_reasoning_effort={json.dumps(reasoning_effort)}",
                ]
            )
            for image_path in image_paths:
                command.extend(["--image", str(image_path)])
            command.append("-")

            result = subprocess.run(
                command,
                input=prompt,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout,
                check=False,
                env=_codex_subprocess_environment(),
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Codex CLI failed")
            response = output_path.read_text(encoding="utf-8").strip()
            if not response:
                raise RuntimeError("Codex CLI returned an empty response")
            return response

    @staticmethod
    def _is_gpt5_model(model_name: str) -> bool:
        name = (model_name or "").lower()
        return "gpt-5" in name or "gpt5" in name

    @retry(
        retry=retry_if_not_exception_type(openai.BadRequestError),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5) | stop_after_delay(DEFAULT_GPT_TIMEOUT),
    )
    def completion_with_backoff(self, **kwargs):
        """Performs a chat completion request with retry/backoff."""
        return self.client.chat.completions.create(**kwargs)

    def query(
        self,
        text_prompt: str,
        image_base64: Optional[
            str | Image.Image | list[str | Image.Image]
        ] = None,
        system_role: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Optional[str]:
        """Queries the GPT model with text and optional image prompts.

        Args:
            text_prompt (str): Main text input.
            image_base64 (Optional[list[str | Image.Image]], optional): List of image base64 strings, file paths, or PIL Images.
            system_role (Optional[str], optional): System-level instructions.
            params (Optional[dict], optional): Additional GPT parameters.

        Returns:
            Optional[str]: Model response content, or None if error.
        """
        if system_role is None:
            system_role = "You are a highly knowledgeable assistant specializing in physics, engineering, and object properties."  # noqa

        if self.provider == "codex":
            try:
                response = self._query_codex(
                    text_prompt=text_prompt,
                    image_base64=image_base64,
                    system_role=system_role,
                    params=params,
                )
            except Exception as e:
                logger.error(f"Error Codex CLI call: {e}")
                response = None

            if self.verbose:
                logger.info(f"Prompt: {text_prompt}")
                logger.info(f"Response: {response}")
            return response

        content_user = [
            {
                "type": "text",
                "text": text_prompt,
            },
        ]

        # Process images if provided
        if image_base64 is not None:
            if not isinstance(image_base64, list):
                image_base64 = [image_base64]
            # Hardcode tmp because of the openrouter can't input multi images.
            if "openrouter" in self.endpoint:
                image_base64 = combine_images_to_grid(image_base64)
            for img in image_base64:
                if isinstance(img, Image.Image):
                    buffer = BytesIO()
                    img.save(buffer, format=img.format or "PNG")
                    buffer.seek(0)
                    image_binary = buffer.read()
                    img = base64.b64encode(image_binary).decode("utf-8")
                elif (
                    len(os.path.splitext(img)) > 1
                    and os.path.splitext(img)[-1].lower() in self.image_formats
                ):
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"Image file not found: {img}")
                    with open(img, "rb") as f:
                        img = base64.b64encode(f.read()).decode("utf-8")

                content_user.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        is_gpt5 = self._is_gpt5_model(self.model_name)
        if is_gpt5:
            # GPT-5.x only supports default temperature/top_p and uses
            # `max_completion_tokens` instead of `max_tokens`.
            payload = {
                "messages": [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": content_user},
                ],
                "max_completion_tokens": GPT5_DEFAULT_MAX_COMPLETION_TOKENS,
                "model": self.model_name,
            }
        else:
            payload = {
                "messages": [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": content_user},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
                "top_p": 0.1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "model": self.model_name,
            }

        if params:
            params = dict(params)
            if is_gpt5:
                # GPT-5.x rejects custom temperature/top_p/penalty/stop and
                # uses `max_completion_tokens` instead of `max_tokens`.
                if (
                    "max_tokens" in params
                    and "max_completion_tokens" not in params
                ):
                    params["max_completion_tokens"] = params.pop("max_tokens")
                for k in (
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                    "max_tokens",
                ):
                    params.pop(k, None)
            payload.update(params)

        response = None
        try:
            response = self.completion_with_backoff(**payload)
            response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error GPTclint {self.endpoint} API call: {e}")
            response = None

        if self.verbose:
            logger.info(f"Prompt: {text_prompt}")
            logger.info(f"Response: {response}")

        return response

    def check_connection(self) -> None:
        """Checks whether the GPT API connection is working.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            if self.provider == "codex":
                response = self._query_codex(
                    text_prompt="Reply with OK.",
                    image_base64=None,
                    system_role="You are a test system.",
                    params=None,
                )
                if not response:
                    raise ConnectionError(
                        "Codex CLI returned an empty response"
                    )
                logger.info("Connection check success.")
                return

            probe_kwargs = dict(
                messages=[
                    {"role": "system", "content": "You are a test system."},
                    {"role": "user", "content": "Hello"},
                ],
                model=self.model_name,
            )
            if self._is_gpt5_model(self.model_name):
                probe_kwargs["max_completion_tokens"] = 100
            else:
                probe_kwargs["temperature"] = 0
                probe_kwargs["max_tokens"] = 100
            response = self.completion_with_backoff(**probe_kwargs)
            response.choices[0].message.content
            logger.info("Connection check success.")
        except Exception:
            raise ConnectionError(
                f"Failed to connect to GPT API at {self.endpoint}, "
                f"please check setting in `{CONFIG_FILE}` and `README`."
            )


with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

settings = _resolve_agent_settings(config)

GPT_CLIENT = GPTclient(
    endpoint=settings["endpoint"],
    api_key=settings["api_key"],
    api_version=settings["api_version"],
    model_name=settings["model_name"],
    check_connection=False,
    timeout=DEFAULT_GPT_TIMEOUT,
    provider=settings["provider"],
)


if __name__ == "__main__":
    response = GPT_CLIENT.query("What is the capital of China?")
    print(f"Response: {response}")
