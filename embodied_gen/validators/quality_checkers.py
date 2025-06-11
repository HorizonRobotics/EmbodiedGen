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
import os

from tqdm import tqdm
from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient
from embodied_gen.utils.process_media import render_asset3d
from embodied_gen.validators.aesthetic_predictor import AestheticPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseChecker:
    def __init__(self, prompt: str = None, verbose: bool = False) -> None:
        self.prompt = prompt
        self.verbose = verbose

    def query(self, *args, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement the query method."
        )

    def __call__(self, *args, **kwargs) -> bool:
        response = self.query(*args, **kwargs)
        if response is None:
            response = "Error when calling gpt api."

        if self.verbose and response != "YES":
            logger.info(response)

        flag = "YES" in response
        response = "YES" if flag else response

        return flag, response

    @staticmethod
    def validate(
        checkers: list["BaseChecker"], images_list: list[list[str]]
    ) -> list:
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
            Refer to the provided multi-view rendering images to evaluate
            whether the geometry of the 3D object asset is complete and
            whether the asset can be placed stably on the ground.
            Return "YES" only if reach the requirments,
            otherwise "NO" and explain the reason very briefly.
            """

    def query(self, image_paths: str) -> str:
        # Hardcode tmp because of the openrouter can't input multi images.
        if "openrouter" in self.gpt_client.endpoint:
            from embodied_gen.utils.process_media import (
                combine_images_to_base64,
            )

            image_paths = combine_images_to_base64(image_paths)

        return self.gpt_client.query(
            text_prompt=self.prompt,
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
            The first image is the original, and the second image is the
            result after segmenting the main object. Evaluate the segmentation
            quality to ensure the main object is clearly segmented without
            significant truncation. Note that the foreground of the object
            needs to be extracted instead of the background.
            Minor imperfections can be ignored. If segmentation is acceptable,
            return "YES" only; otherwise, return "NO" with
            very brief explanation.
            """

    def query(self, image_paths: list[str]) -> str:
        if len(image_paths) != 2:
            raise ValueError(
                "ImageSegChecker requires exactly two images: [raw_image, seg_image]."  # noqa
            )
        # Hardcode tmp because of the openrouter can't input multi images.
        if "openrouter" in self.gpt_client.endpoint:
            from embodied_gen.utils.process_media import (
                combine_images_to_base64,
            )

            image_paths = combine_images_to_base64(image_paths)

        return self.gpt_client.query(
            text_prompt=self.prompt,
            image_base64=image_paths,
        )


class ImageAestheticChecker(BaseChecker):
    """A class for evaluating the aesthetic quality of images.

    Attributes:
        clip_model_dir (str): Path to the CLIP model directory.
        sac_model_path (str): Path to the aesthetic predictor model weights.
        thresh (float): Threshold above which images are considered aesthetically acceptable.
        verbose (bool): Whether to print detailed log messages.
        predictor (AestheticPredictor): The model used to predict aesthetic scores.
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


if __name__ == "__main__":
    geo_checker = MeshGeoChecker(GPT_CLIENT)
    seg_checker = ImageSegChecker(GPT_CLIENT)
    aesthetic_checker = ImageAestheticChecker()

    checkers = [geo_checker, seg_checker, aesthetic_checker]

    output_root = "outputs/test_gpt"

    fails = []
    for idx in tqdm(range(150)):
        mesh_path = f"outputs/imageto3d/demo_objects/cups/sample_{idx}/sample_{idx}.obj"  # noqa
        if not os.path.exists(mesh_path):
            continue
        image_paths = render_asset3d(
            mesh_path,
            f"{output_root}/{idx}",
            num_images=8,
            elevation=(30, -30),
            distance=5.5,
        )

        for cid, checker in enumerate(checkers):
            if isinstance(checker, ImageSegChecker):
                images = [
                    f"outputs/imageto3d/demo_objects/cups/sample_{idx}/sample_{idx}_raw.png",  # noqa
                    f"outputs/imageto3d/demo_objects/cups/sample_{idx}/sample_{idx}_cond.png",  # noqa
                ]
            else:
                images = image_paths
            result, info = checker(images)
            logger.info(
                f"Checker {checker.__class__.__name__}: {result}, {info}, mesh {mesh_path}"  # noqa
            )

            if result is False:
                fails.append((idx, cid, info))

        break
