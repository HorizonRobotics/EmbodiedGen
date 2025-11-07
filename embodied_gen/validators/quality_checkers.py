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
import random

import json_repair
from PIL import Image
from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient
from embodied_gen.validators.aesthetic_predictor import AestheticPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "MeshGeoChecker",
    "ImageSegChecker",
    "ImageAestheticChecker",
    "SemanticConsistChecker",
    "TextGenAlignChecker",
    "PanoImageGenChecker",
    "PanoHeightEstimator",
    "PanoImageOccChecker",
]


class BaseChecker:
    """Base class for quality checkers using GPT clients.

    Provides a common interface for querying and validating responses.
    Subclasses must implement the `query` method.

    Attributes:
        prompt (str): The prompt used for queries.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(self, prompt: str = None, verbose: bool = False) -> None:
        self.prompt = prompt
        self.verbose = verbose

    def query(self, *args, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement the query method."
        )

    def __call__(self, *args, **kwargs) -> tuple[bool, str]:
        response = self.query(*args, **kwargs)
        if self.verbose:
            logger.info(response)

        if response is None:
            flag = None
            response = (
                "Error when calling GPT api, check config in "
                "`embodied_gen/utils/gpt_config.yaml` or net connection."
            )
        else:
            flag = "YES" in response
            response = "YES" if flag else response

        return flag, response

    @staticmethod
    def validate(
        checkers: list["BaseChecker"], images_list: list[list[str]]
    ) -> list:
        """Validates a list of checkers against corresponding image lists.

        Args:
            checkers (list[BaseChecker]): List of checker instances.
            images_list (list[list[str]]): List of image path lists.

        Returns:
            list: Validation results with overall outcome.
        """
        assert len(checkers) == len(images_list)
        results = []
        overall_result = True
        for checker, images in zip(checkers, images_list):
            qa_flag, qa_info = checker(images)
            if isinstance(qa_info, str):
                qa_info = qa_info.replace("\n", ".")
            results.append([checker.__class__.__name__, qa_info])
            if qa_flag is False:
                overall_result = False

        results.append(["overall", "YES" if overall_result else "NO"])

        return results


class MeshGeoChecker(BaseChecker):
    """A geometry quality checker for 3D mesh assets using GPT-based reasoning.

    This class leverages a multi-modal GPT client to analyze rendered images
    of a 3D object and determine if its geometry is complete.

    Attributes:
        gpt_client (GPTclient): The GPT client used for multi-modal querying.
        prompt (str): The prompt sent to the GPT model. If not provided, a default one is used.
        verbose (bool): Whether to print debug information during evaluation.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        if self.prompt is None:
            self.prompt = """
            You are an expert in evaluating the geometry quality of generated 3D asset.
            You will be given rendered views of a generated 3D asset, type {}, with black background.
            Your task is to evaluate the quality of the 3D asset generation,
            including geometry, structure, and appearance, based on the rendered views.
            Criteria:
            - Is the object in the image a single, complete, and well-formed instance,
                without truncation, missing parts, overlapping duplicates, or redundant geometry?
            - Minor flaws, asymmetries, or simplifications (e.g., less detail on sides or back,
                soft edges) are acceptable if the object is structurally sound and recognizable.
            - Only evaluate geometry. Do not assess texture quality.
            - The asset should not contain any unrelated elements, such as
                ground planes, platforms, or background props (e.g., paper, flooring).

            If all the above criteria are met, return "YES". Otherwise, return
                "NO" followed by a brief explanation (no more than 20 words).

            Example:
            Images show a yellow cup standing on a flat white plane -> NO
            -> Response: NO: extra white surface under the object.
            Image shows a chair with simplified back legs and soft edges → YES
            """

    def query(
        self, image_paths: list[str | Image.Image], text: str = "unknown"
    ) -> str:
        input_prompt = self.prompt.format(text)

        return self.gpt_client.query(
            text_prompt=input_prompt,
            image_base64=image_paths,
        )


class ImageSegChecker(BaseChecker):
    """A segmentation quality checker for 3D assets using GPT-based reasoning.

    This class compares an original image with its segmented version to
    evaluate whether the segmentation successfully isolates the main object
    with minimal truncation and correct foreground extraction.

    Attributes:
        gpt_client (GPTclient): GPT client used for multi-modal image analysis.
        prompt (str): The prompt used to guide the GPT model for evaluation.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        if self.prompt is None:
            self.prompt = """
            Task: Evaluate the quality of object segmentation between two images:
                the first is the original, the second is the segmented result.

            Criteria:
            - The main foreground object should be clearly extracted (not the background).
            - The object must appear realistic, with reasonable geometry and color.
            - The object should be geometrically complete — no missing, truncated, or cropped parts.
            - The object must be centered, with a margin on all sides.
            - Ignore minor imperfections (e.g., small holes or fine edge artifacts).

            Output Rules:
            If segmentation is acceptable, respond with "YES" (and nothing else).
            If not acceptable, respond with "NO", followed by a brief reason (max 20 words).
            """

    def query(self, image_paths: list[str]) -> str:
        if len(image_paths) != 2:
            raise ValueError(
                "ImageSegChecker requires exactly two images: [raw_image, seg_image]."  # noqa
            )

        return self.gpt_client.query(
            text_prompt=self.prompt,
            image_base64=image_paths,
        )


class ImageAestheticChecker(BaseChecker):
    """Evaluates the aesthetic quality of images using a CLIP-based predictor.

    Attributes:
        clip_model_dir (str): Path to the CLIP model directory.
        sac_model_path (str): Path to the aesthetic predictor model weights.
        thresh (float): Threshold above which images are considered aesthetically acceptable.
        verbose (bool): Whether to print detailed log messages.
        predictor (AestheticPredictor): The model used to predict aesthetic scores.

    Example:
        ```py
        from embodied_gen.validators.quality_checkers import ImageAestheticChecker
        checker = ImageAestheticChecker(thresh=4.5)
        flag, score = checker(["image1.png", "image2.png"])
        print("Aesthetic OK:", flag, "Score:", score)
        ```
    """

    def __init__(
        self,
        clip_model_dir: str = None,
        sac_model_path: str = None,
        thresh: float = 4.50,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.clip_model_dir = clip_model_dir
        self.sac_model_path = sac_model_path
        self.thresh = thresh
        self.predictor = AestheticPredictor(clip_model_dir, sac_model_path)

    def query(self, image_paths: list[str]) -> float:
        scores = [self.predictor.predict(img_path) for img_path in image_paths]
        return sum(scores) / len(scores)

    def __call__(self, image_paths: list[str], **kwargs) -> bool:
        avg_score = self.query(image_paths)
        if self.verbose:
            logger.info(f"Average aesthetic score: {avg_score}")
        return avg_score > self.thresh, avg_score


class SemanticConsistChecker(BaseChecker):
    """Checks semantic consistency between text descriptions and segmented images.

    Uses GPT to evaluate if the image matches the text in object type, geometry, and color.

    Attributes:
        gpt_client (GPTclient): GPT client for queries.
        prompt (str): Prompt for consistency evaluation.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        if self.prompt is None:
            self.prompt = """
            You are an expert in image-text consistency assessment.
            You will be given:
            - A short text description of an object.
            - An segmented image of the same object with the background removed.

            Criteria:
            - The image must visually match the text description in terms of object type, structure, geometry, and color.
            - The object must appear realistic, with reasonable geometry (e.g., a table must have a stable number
                of legs with a reasonable distribution. Count the number of legs visible in the image. (strict) For tables,
                fewer than four legs or if the legs are unevenly distributed, are not allowed. Do not assume
                hidden legs unless they are clearly visible.)
            - Geometric completeness is required: the object must not have missing, truncated, or cropped parts.
            - The image must contain exactly one object. Multiple distinct objects (e.g. multiple pens) are not allowed.
                A single composite object (e.g., a chair with legs) is acceptable.
            - The object should be shown from a slightly angled (three-quarter) perspective,
                not a flat, front-facing view showing only one surface.

            Instructions:
            - If all criteria are met, return `"YES"`.
            - Otherwise, return "NO" with a brief explanation (max 20 words).

            Respond in exactly one of the following formats:
            YES
            or
            NO: brief explanation.

            Input:
            {}
            """

    def query(self, text: str, image: list[Image.Image | str]) -> str:

        return self.gpt_client.query(
            text_prompt=self.prompt.format(text),
            image_base64=image,
        )


class TextGenAlignChecker(BaseChecker):
    """Evaluates alignment between text prompts and generated 3D asset images.

    Assesses if the rendered images match the text description in category and geometry.

    Attributes:
        gpt_client (GPTclient): GPT client for queries.
        prompt (str): Prompt for alignment evaluation.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        if self.prompt is None:
            self.prompt = """
            You are an expert in evaluating the quality of generated 3D assets.
            You will be given:
            - A text description of an object: TEXT
            - Rendered views of the generated 3D asset.

            Your task is to:
            1. Determine whether the generated 3D asset roughly reflects the object class
                or a semantically adjacent category described in the text.
            2. Evaluate the geometry quality of the 3D asset generation based on the rendered views.

            Criteria:
            - Determine if the generated 3D asset belongs to the text described or a similar category.
            - Focus on functional similarity: if the object serves the same general
                purpose (e.g., writing, placing items), it should be accepted.
            - Is the geometry complete and well-formed, with no missing parts,
            distortions, visual artifacts, or redundant structures?
            - Does the number of object instances match the description?
                There should be only one object unless otherwise specified.
            - Minor flaws in geometry or texture are acceptable, high tolerance for texture quality defects.
            - Minor simplifications in geometry or texture (e.g. soft edges, less detail)
                are acceptable if the object is still recognizable.
            - The asset should not contain any unrelated elements, such as
                ground planes, platforms, or background props (e.g., paper, flooring).

            Example:
            Text: "yellow cup"
            Image: shows a yellow cup standing on a flat white plane -> NO: extra surface under the object.

            Instructions:
            - If the quality of generated asset is acceptable and faithfully represents the text, return "YES".
            - Otherwise, return "NO" followed by a brief explanation (no more than 20 words).

            Respond in exactly one of the following formats:
            YES
            or
            NO: brief explanation

            Input:
            Text description: {}
            """

    def query(self, text: str, image: list[Image.Image | str]) -> str:

        return self.gpt_client.query(
            text_prompt=self.prompt.format(text),
            image_base64=image,
        )


class PanoImageGenChecker(BaseChecker):
    """A checker class that validates the quality and realism of generated panoramic indoor images.

    Attributes:
        gpt_client (GPTclient): A GPT client instance used to query for image validation.
        prompt (str): The instruction prompt passed to the GPT model. If None, a default prompt is used.
        verbose (bool): Whether to print internal processing information for debugging.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        if self.prompt is None:
            self.prompt = """
            You are a panoramic image analyzer specializing in indoor room structure validation.

            Given a generated panoramic image, assess if it meets all the criteria:
            - Floor Space: ≥30 percent of the floor is free of objects or obstructions.
            - Visual Clarity: Floor, walls, and ceiling are clear, with no distortion, blur, noise.
            - Structural Continuity: Surfaces form plausible, continuous geometry
                without breaks, floating parts, or abrupt cuts.
            - Spatial Completeness: Full 360° coverage without missing areas,
                seams, gaps, or stitching artifacts.
            Instructions:
            - If all criteria are met, reply with "YES".
            - Otherwise, reply with "NO: <brief explanation>" (max 20 words).

            Respond exactly as:
            "YES"
            or
            "NO: brief explanation."
            """

    def query(self, image_paths: str | Image.Image) -> str:

        return self.gpt_client.query(
            text_prompt=self.prompt,
            image_base64=image_paths,
        )


class PanoImageOccChecker(BaseChecker):
    """Checks for physical obstacles in the bottom-center region of a panoramic image.

    This class crops a specified region from the input panoramic image and uses
    a GPT client to determine whether any physical obstacles there.

    Args:
        gpt_client (GPTclient): The GPT-based client used for visual reasoning.
        box_hw (tuple[int, int]): The height and width of the crop box.
        prompt (str, optional): Custom prompt for the GPT client. Defaults to a predefined one.
        verbose (bool, optional): Whether to print verbose logs. Defaults to False.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        box_hw: tuple[int, int],
        prompt: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        self.box_hw = box_hw
        if self.prompt is None:
            self.prompt = """
            This image is a cropped region from the bottom-center of a panoramic view.
            Please determine whether there is any obstacle present — such as furniture, tables, or other physical objects.
            Ignore floor textures, rugs, carpets, shadows, and lighting effects — they do not count as obstacles.
            Only consider real, physical objects that could block walking or movement.

            Instructions:
            - If there is no obstacle, reply: "YES".
            - Otherwise, reply: "NO: <brief explanation>" (max 20 words).

            Respond exactly as:
            "YES"
            or
            "NO: brief explanation."
            """

    def query(self, image_paths: str | Image.Image) -> str:
        if isinstance(image_paths, str):
            image_paths = Image.open(image_paths)

        w, h = image_paths.size
        image_paths = image_paths.crop(
            (
                (w - self.box_hw[1]) // 2,
                h - self.box_hw[0],
                (w + self.box_hw[1]) // 2,
                h,
            )
        )

        return self.gpt_client.query(
            text_prompt=self.prompt,
            image_base64=image_paths,
        )


class PanoHeightEstimator(object):
    """Estimate the real ceiling height of an indoor space from a 360° panoramic image.

    Attributes:
        gpt_client (GPTclient): The GPT client used to perform image-based reasoning and return height estimates.
        default_value (float): The fallback height in meters if parsing the GPT output fails.
        prompt (str): The textual instruction used to guide the GPT model for height estimation.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        default_value: float = 3.5,
    ) -> None:
        self.gpt_client = gpt_client
        self.default_value = default_value
        self.prompt = """
        You are an expert in building height estimation and panoramic image analysis.
        Your task is to analyze a 360° indoor panoramic image and estimate the **actual height** of the space in meters.

        Consider the following visual cues:
        1. Ceiling visibility and reference objects (doors, windows, furniture, appliances).
        2. Floor features or level differences.
        3. Room type (e.g., residential, office, commercial).
        4. Object-to-ceiling proportions (e.g., height of doors relative to ceiling).
        5. Architectural elements (e.g., chandeliers, shelves, kitchen cabinets).

        Input: A full 360° panoramic indoor photo.
        Output: A single number in meters representing the estimated room height. Only return the number (e.g., `3.2`)
        """

    def __call__(self, image_paths: str | Image.Image) -> float:
        result = self.gpt_client.query(
            text_prompt=self.prompt,
            image_base64=image_paths,
        )
        try:
            result = float(result.strip())
        except Exception as e:
            logger.error(
                f"Parser error: failed convert {result} to float, {e}, use default value {self.default_value}."
            )
            result = self.default_value

        return result


class SemanticMatcher(BaseChecker):
    """Matches query text to semantically similar scene descriptions.

    Uses GPT to find the most similar scene IDs from a dictionary.

    Attributes:
        gpt_client (GPTclient): GPT client for queries.
        prompt (str): Prompt for semantic matching.
        verbose (bool): Whether to enable verbose logging.
        seed (int): Random seed for selection.
    """

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str = None,
        verbose: bool = False,
        seed: int = None,
    ) -> None:
        super().__init__(prompt, verbose)
        self.gpt_client = gpt_client
        self.seed = seed
        random.seed(seed)
        if self.prompt is None:
            self.prompt = """
            You are an expert in semantic similarity and scene retrieval.
            You will be given:
            - A dictionary where each key is a scene ID, and each value is a scene description.
            - A query text describing a target scene.

            Your task:
            return_num = 2
            - Find the <return_num> most semantically similar scene IDs to the query text.
            - If there are fewer than <return_num> distinct relevant matches, repeat the closest ones to make a list of <return_num>.
            - Only output the list of <return_num> scene IDs, sorted from most to less similar.
            - Do NOT use markdown, JSON code blocks, or any formatting syntax, only return a plain list like ["id1", ...].
            - The returned scene ID must exist in the dictionary and be in exactly the same format. For example,
                if the key in the dictionary is "scene_0040", return "scene_0040"; if it is "scene_040", return "scene_040".

            Input example:
            Dictionary:
            "{{
            "t_scene_0008": "A study room with full bookshelves and a lamp in the corner.",
            "t_scene_019": "A child's bedroom with pink walls and a small desk.",
            "t_scene_020": "A living room with a wooden floor.",
            "t_scene_021": "A living room with toys scattered on the floor.",
            ...
            "t_scene_office_0001": "A very spacious, modern open-plan office with wide desks and no people, panoramic view."
            }}"
            Text:
            "A traditional indoor room"
            Output:
            '["t_scene_office_0001", ...]'

            Input:
            Dictionary:
            {context}
            Text:
            {text}
            Output:
            <topk_key_list>
            """

    def query(
        self, text: str, context: dict, rand: bool = True, params: dict = None
    ) -> str:
        """Queries for semantically similar scene IDs.

        Args:
            text (str): Query text.
            context (dict): Dictionary of scene descriptions.
            rand (bool, optional): Whether to randomly select from top matches.
            params (dict, optional): Additional GPT parameters.

        Returns:
            str: Matched scene ID.
        """
        match_list = self.gpt_client.query(
            self.prompt.format(context=context, text=text),
            params=params,
        )
        match_list = json_repair.loads(match_list)
        result = random.choice(match_list) if rand else match_list[0]

        return result


def test_semantic_matcher(
    bg_file: str = "outputs/bg_scenes/scene_list.txt",
):
    scene_dict = {}
    with open(bg_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            scene_id, desc = line.split(":", 1)
            scene_dict[scene_id.strip()] = desc.strip()

    office_scene = scene_dict.get("t_scene_office_001")
    text = "bright kitchen"
    SCENE_MATCHER = SemanticMatcher(GPT_CLIENT)
    # gpt_params = {
    #     "temperature": 0.8,
    #     "max_tokens": 500,
    #     "top_p": 0.8,
    #     "frequency_penalty": 0.3,
    #     "presence_penalty": 0.3,
    # }
    gpt_params = None
    match_key = SCENE_MATCHER.query(text, str(scene_dict), params=gpt_params)
    print(match_key, ",", scene_dict[match_key])


if __name__ == "__main__":
    test_semantic_matcher()
