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


import argparse
import json
import logging
import os
import re

import json_repair
from embodied_gen.utils.enum import (
    LayoutInfo,
    RobotItemEnum,
    Scene3DItemEnum,
    SpatialRelationEnum,
)
from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient
from embodied_gen.utils.process_media import SceneTreeVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "LayoutDesigner",
    "LAYOUT_DISASSEMBLER",
    "LAYOUT_GRAPHER",
    "LAYOUT_DESCRIBER",
]


DISTRACTOR_NUM = 3  # Maximum number of distractor objects allowed
LAYOUT_DISASSEMBLE_PROMPT = f"""
    You are an intelligent 3D scene planner. Given a natural language
    description of a robotic task, output a structured description of
    an interactive 3D scene.

    The output must include the following fields:
    - task: A high-level task type (e.g., "single-arm pick",
        "dual-arm grasping", "pick and place", "object sorting").
    - {Scene3DItemEnum.ROBOT}: The name or type of robot involved. If not mentioned,
        use {RobotItemEnum.FRANKA} as default.
    - {Scene3DItemEnum.BACKGROUND}: The room or indoor environment where the task happens
        (e.g., Kitchen, Bedroom, Living Room, Workshop, Office).
    - {Scene3DItemEnum.CONTEXT}: A indoor object involved in the manipulation
        (e.g., Table, Shelf, Desk, Bed, Cabinet).
    - {Scene3DItemEnum.MANIPULATED_OBJS}: The main object(s) that the robot directly interacts with.
    - {Scene3DItemEnum.DISTRACTOR_OBJS}: Other objects that naturally belong to the scene but are not part of the main task.

    Constraints:
    - The {Scene3DItemEnum.BACKGROUND} must logically match the described task.
    - The {Scene3DItemEnum.CONTEXT} must fit within the {Scene3DItemEnum.BACKGROUND}. (e.g., a bedroom may include a table or bed, but not a workbench.)
    - The {Scene3DItemEnum.CONTEXT} must be a concrete indoor object, such as a "table",
        "shelf", "desk", or "bed". It must not be an abstract concept (e.g., "area", "space", "zone")
        or structural surface (e.g., "floor", "ground"). If the input describes an interaction near
        the floor or vague space, you must infer a plausible object like a "table", "cabinet", or "storage box" instead.
    - {Scene3DItemEnum.MANIPULATED_OBJS} and {Scene3DItemEnum.DISTRACTOR_OBJS} objects must be plausible,
        and semantically compatible with the {Scene3DItemEnum.CONTEXT} and {Scene3DItemEnum.BACKGROUND}.
    - {Scene3DItemEnum.DISTRACTOR_OBJS} must not confuse or overlap with the manipulated objects.
    - {Scene3DItemEnum.DISTRACTOR_OBJS} number limit: {DISTRACTOR_NUM} distractors maximum.
    - All {Scene3DItemEnum.BACKGROUND} are limited to indoor environments.
    - {Scene3DItemEnum.MANIPULATED_OBJS} and {Scene3DItemEnum.DISTRACTOR_OBJS} are rigid bodies and not include flexible objects.
    - {Scene3DItemEnum.MANIPULATED_OBJS} and {Scene3DItemEnum.DISTRACTOR_OBJS} must be common
        household or office items or furniture, not abstract concepts, not too small like needle.
    - If the input includes a plural or grouped object (e.g., "pens", "bottles", "plates", "fruit"),
        you must decompose it into multiple individual instances (e.g., ["pen", "pen"], ["apple", "pear"]).
    - Containers that hold objects (e.g., "bowl of apples", "box of tools") must
        be separated into individual items (e.g., ["bowl", "apple", "apple"]).
    - Do not include transparent objects such as "glass", "plastic", etc.
    - The output must be in compact JSON format and use Markdown syntax, just like the output in the example below.

    Examples:

    Input:
    "Pick up the marker from the table and put it in the bowl robot {RobotItemEnum.UR5}."
    Output:
    ```json
    {{
        "task_desc": "Pick up the marker from the table and put it in the bowl.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.UR5}",
        "{Scene3DItemEnum.BACKGROUND}": "kitchen",
        "{Scene3DItemEnum.CONTEXT}": "table",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["marker"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["mug", "notebook", "bowl"]
    }}
    ```

    Input:
    "Put the rubik's cube on the top of the shelf."
    Output:
    ```json
    {{
        "task_desc": "Put the rubik's cube on the top of the shelf.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.FRANKA}",
        "{Scene3DItemEnum.BACKGROUND}": "bedroom",
        "{Scene3DItemEnum.CONTEXT}": "shelf",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["rubik's cube"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["pen", "cup", "toy car"]
    }}
    ```

    Input:
    "Remove all the objects from the white basket and put them on the table."
    Output:
    ```json
    {{
        "task_desc": "Remove all the objects from the white basket and put them on the table, robot {RobotItemEnum.PIPER}.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.PIPER}",
        "{Scene3DItemEnum.BACKGROUND}": "office",
        "{Scene3DItemEnum.CONTEXT}": "table",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["banana", "mobile phone"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["plate", "white basket"]
    }}
    ```

    Input:
    "Pick up the rope on the chair and put it in the box."
    Output:
    ```json
    {{
        "task_desc": "Pick up the rope on the chair and put it in the box, robot {RobotItemEnum.FRANKA}.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.FRANKA}",
        "{Scene3DItemEnum.BACKGROUND}": "living room",
        "{Scene3DItemEnum.CONTEXT}": "chair",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["rope", "box"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["magazine"]
    }}
    ```

    Input:
    "Pick up the seal tape and plastic from the counter and put them in the open drawer and close it."
    Output:
    ```json
    {{
        "task_desc": "Pick up the seal tape and plastic from the counter and put them in the open drawer and close it.",
        "task": "pick and place",
        "robot": "franka",
        "background": "kitchen",
        "context": "counter",
        "manipulated_objs": ["seal tape", "plastic", "opened drawer"],
        "distractor_objs": ["scissors"]
    }}
    ```

    Input:
    "Put the pens in the grey bowl."
    Output:
    ```json
    {{
        "task_desc": "Put the pens in the grey bowl.",
        "task": "pick and place",
        "robot": "franka",
        "background": "office",
        "context": "table",
        "manipulated_objs": ["pen", "pen", "grey bowl"],
        "distractor_objs": ["notepad", "cup"]
    }}
    ```

"""


LAYOUT_HIERARCHY_PROMPT = f"""
    You are a 3D scene layout reasoning expert.
    Your task is to generate a spatial relationship dictionary in multiway tree
    that describes how objects are arranged in a 3D environment
    based on a given task description and object list.

    Input in JSON format containing the task description, task type,
    {Scene3DItemEnum.ROBOT}, {Scene3DItemEnum.BACKGROUND}, {Scene3DItemEnum.CONTEXT},
    and a list of objects, including {Scene3DItemEnum.MANIPULATED_OBJS} and {Scene3DItemEnum.DISTRACTOR_OBJS}.

    ### Supported Spatial Relations:
    - "{SpatialRelationEnum.ON}": The child object bottom is directly on top of the parent object top.
    - "{SpatialRelationEnum.INSIDE}": The child object is inside the context object.
    - "{SpatialRelationEnum.IN}": The {Scene3DItemEnum.ROBOT} in the {Scene3DItemEnum.BACKGROUND}.
    - "{SpatialRelationEnum.FLOOR}": The child object bottom is on the floor of the {Scene3DItemEnum.BACKGROUND}.

    ### Rules:
    - The {Scene3DItemEnum.CONTEXT} object must be "{SpatialRelationEnum.FLOOR}" the {Scene3DItemEnum.BACKGROUND}.
    - {Scene3DItemEnum.MANIPULATED_OBJS} and {Scene3DItemEnum.DISTRACTOR_OBJS} must be either
        "{SpatialRelationEnum.ON}" or "{SpatialRelationEnum.INSIDE}" the {Scene3DItemEnum.CONTEXT}
    - Or "{SpatialRelationEnum.FLOOR}" {Scene3DItemEnum.BACKGROUND}.
    - Use "{SpatialRelationEnum.INSIDE}" only if the parent is a container-like object (e.g., shelf, rack, cabinet).
    - Do not define relationship edges between objects, only for the child and parent nodes.
    - {Scene3DItemEnum.ROBOT} must "{SpatialRelationEnum.IN}" the {Scene3DItemEnum.BACKGROUND}.
    - Ensure that each object appears only once in the layout tree, and its spatial relationship is defined with only one parent.
    - Ensure a valid multiway tree structure with a maximum depth of 2 levels suitable for a 3D scene layout representation.
    - Only output the final output in JSON format, using Markdown syntax as in examples.

    ### Example
    Input:
    {{
        "task_desc": "Pick up the marker from the table and put it in the bowl.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.FRANKA}",
        "{Scene3DItemEnum.BACKGROUND}": "kitchen",
        "{Scene3DItemEnum.CONTEXT}": "table",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["marker", "bowl"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["mug", "chair"]
    }}
    Intermediate Think:
        table {SpatialRelationEnum.FLOOR} kitchen
        chair {SpatialRelationEnum.FLOOR} kitchen
        {RobotItemEnum.FRANKA} {SpatialRelationEnum.IN} kitchen
        marker {SpatialRelationEnum.ON} table
        bowl {SpatialRelationEnum.ON} table
        mug {SpatialRelationEnum.ON} table
    Final Output:
    ```json
    {{
        "kitchen": [
            ["table", "{SpatialRelationEnum.FLOOR}"],
            ["chair", "{SpatialRelationEnum.FLOOR}"],
            ["{RobotItemEnum.FRANKA}", "{SpatialRelationEnum.IN}"]
        ],
        "table": [
            ["marker", "{SpatialRelationEnum.ON}"],
            ["bowl", "{SpatialRelationEnum.ON}"],
            ["mug", "{SpatialRelationEnum.ON}"]
        ]
    }}
    ```

    Input:
    {{
        "task_desc": "Put the marker on top of the book.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.UR5}",
        "{Scene3DItemEnum.BACKGROUND}": "office",
        "{Scene3DItemEnum.CONTEXT}": "desk",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["marker", "book"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["pen holder", "notepad"]
    }}
    Intermediate Think:
        desk {SpatialRelationEnum.FLOOR} office
        {RobotItemEnum.UR5} {SpatialRelationEnum.IN} office
        marker {SpatialRelationEnum.ON} desk
        book {SpatialRelationEnum.ON} desk
        pen holder {SpatialRelationEnum.ON} desk
        notepad {SpatialRelationEnum.ON} desk
    Final Output:
    ```json
    {{
        "office": [
            ["desk", "{SpatialRelationEnum.FLOOR}"],
            ["{RobotItemEnum.UR5}", "{SpatialRelationEnum.IN}"]
        ],
        "desk": [
            ["marker", "{SpatialRelationEnum.ON}"],
            ["book", "{SpatialRelationEnum.ON}"],
            ["pen holder", "{SpatialRelationEnum.ON}"],
            ["notepad", "{SpatialRelationEnum.ON}"]
        ]
    }}
    ```

    Input:
    {{
        "task_desc": "Put the rubik's cube on the top of the shelf.",
        "task": "pick and place",
        "{Scene3DItemEnum.ROBOT}": "{RobotItemEnum.UR5}",
        "{Scene3DItemEnum.BACKGROUND}": "bedroom",
        "{Scene3DItemEnum.CONTEXT}": "shelf",
        "{Scene3DItemEnum.MANIPULATED_OBJS}": ["rubik's cube"],
        "{Scene3DItemEnum.DISTRACTOR_OBJS}": ["toy car", "pen"]
    }}
    Intermediate Think:
        shelf {SpatialRelationEnum.FLOOR} bedroom
        {RobotItemEnum.UR5} {SpatialRelationEnum.IN} bedroom
        rubik's cube {SpatialRelationEnum.INSIDE} shelf
        toy car {SpatialRelationEnum.INSIDE} shelf
        pen {SpatialRelationEnum.INSIDE} shelf
    Final Output:
    ```json
    {{
        "bedroom": [
            ["shelf", "{SpatialRelationEnum.FLOOR}"],
            ["{RobotItemEnum.UR5}", "{SpatialRelationEnum.IN}"]
        ],
        "shelf": [
            ["rubik's cube", "{SpatialRelationEnum.INSIDE}"],
            ["toy car", "{SpatialRelationEnum.INSIDE}"],
            ["pen", "{SpatialRelationEnum.INSIDE}"]
        ]
    }}
    ```

    Input:
    {{
        "task_desc": "Put the marker in the cup on the counter.",
        "task": "pick and place",
        "robot": "franka",
        "background": "kitchen",
        "context": "counter",
        "manipulated_objs": ["marker", "cup"],
        "distractor_objs": ["plate", "spoon"]
    }}
    Intermediate Think:
        counter {SpatialRelationEnum.FLOOR} kitchen
        {RobotItemEnum.FRANKA} {SpatialRelationEnum.IN} kitchen
        marker {SpatialRelationEnum.ON} counter
        cup {SpatialRelationEnum.ON} counter
        plate {SpatialRelationEnum.ON} counter
        spoon {SpatialRelationEnum.ON} counter
    Final Output:
    ```json
    {{
        "kitchen": [
            ["counter", "{SpatialRelationEnum.FLOOR}"],
            ["{RobotItemEnum.FRANKA}", "{SpatialRelationEnum.IN}"]
        ],
        "counter": [
            ["marker", "{SpatialRelationEnum.ON}"],
            ["cup", "{SpatialRelationEnum.ON}"],
            ["plate", "{SpatialRelationEnum.ON}"],
            ["spoon", "{SpatialRelationEnum.ON}"]
        ]
    }}
    ```
"""


LAYOUT_DESCRIBER_PROMPT = """
    You are a 3D asset style descriptor.

    Given a task description and a dictionary where the key is the object content and
    the value is the object type, output a JSON dictionary with each object paired
    with a concise, styled visual description suitable for 3D asset generation.

    Generation Guidelines:
    - For each object, brainstorm multiple style candidates before selecting the final
        description. Vary phrasing, material, texture, color, and spatial details.
    - Each description must be a maximum of 15 words, including color, style, materials.
    - Descriptions should be visually grounded, specific, and reflect surface texture and structure.
    - For objects marked as "context", explicitly mention the object is standalone, has an empty top.
    - Use rich style descriptors: e.g., "scratched brown wooden desk" etc.
    - Ensure all object styles align with the task's overall context and environment.

    Format your output in JSON like the example below.

    Example Input:
    "Pick up the rope on the chair and put it in the box. {'living room': 'background', 'chair': 'context',
        'rope': 'manipulated_objs', 'box': 'manipulated_objs', 'magazine': 'distractor_objs'}"

    Example Output:
    ```json
    {
        "living room": "modern cozy living room with soft sunlight and light grey carpet",
        "chair": "standalone dark oak chair with no surroundings and clean empty seat",
        "rope": "twisted hemp rope with rough fibers and dusty beige texture",
        "box": "slightly crumpled cardboard box with open flaps and brown textured surface",
        "magazine": "celebrity magazine with glossy red cover and large bold title"
    }
    ```
"""


class LayoutDesigner(object):
    def __init__(
        self,
        gpt_client: GPTclient,
        system_prompt: str,
        verbose: bool = False,
    ) -> None:
        self.prompt = system_prompt.strip()
        self.verbose = verbose
        self.gpt_client = gpt_client

    def query(self, prompt: str, params: dict = None) -> str:
        full_prompt = self.prompt + f"\n\nInput:\n\"{prompt}\""

        response = self.gpt_client.query(
            text_prompt=full_prompt,
            params=params,
        )

        if self.verbose:
            logger.info(f"Response: {response}")

        return response

    def format_response(self, response: str) -> dict:
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
        try:
            output = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error: {e}, failed to parse JSON response: {response}"
            )

        return output

    def format_response_repair(self, response: str) -> dict:
        return json_repair.loads(response)

    def save_output(self, output: dict, save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)

    def __call__(
        self, prompt: str, save_path: str = None, params: dict = None
    ) -> dict | str:
        response = self.query(prompt, params=params)
        output = self.format_response_repair(response)
        self.save_output(output, save_path) if save_path else None

        return output


LAYOUT_DISASSEMBLER = LayoutDesigner(
    gpt_client=GPT_CLIENT, system_prompt=LAYOUT_DISASSEMBLE_PROMPT
)
LAYOUT_GRAPHER = LayoutDesigner(
    gpt_client=GPT_CLIENT, system_prompt=LAYOUT_HIERARCHY_PROMPT
)
LAYOUT_DESCRIBER = LayoutDesigner(
    gpt_client=GPT_CLIENT, system_prompt=LAYOUT_DESCRIBER_PROMPT
)


def build_scene_layout(
    task_desc: str, output_path: str = None, gpt_params: dict = None
) -> LayoutInfo:
    layout_relation = LAYOUT_DISASSEMBLER(task_desc, params=gpt_params)
    layout_tree = LAYOUT_GRAPHER(layout_relation, params=gpt_params)
    object_mapping = Scene3DItemEnum.object_mapping(layout_relation)
    obj_prompt = f'{layout_relation["task_desc"]} {object_mapping}'
    objs_desc = LAYOUT_DESCRIBER(obj_prompt, params=gpt_params)
    layout_info = LayoutInfo(
        layout_tree, layout_relation, objs_desc, object_mapping
    )

    if output_path is not None:
        visualizer = SceneTreeVisualizer(layout_info)
        visualizer.render(save_path=output_path)
        logger.info(f"Scene hierarchy tree saved to {output_path}")

    return layout_info


def parse_args():
    parser = argparse.ArgumentParser(description="3D Scene Layout Designer")
    parser.add_argument(
        "--task_desc",
        type=str,
        default="Put the apples on the table on the plate",
        help="Natural language description of the robotic task",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="outputs/layout_tree",
        help="Path to save the layout output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from embodied_gen.utils.enum import LayoutInfo
    from embodied_gen.utils.process_media import SceneTreeVisualizer

    args = parse_args()
    params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.5,
    }
    layout_relation = LAYOUT_DISASSEMBLER(args.task_desc, params=params)
    layout_tree = LAYOUT_GRAPHER(layout_relation, params=params)

    object_mapping = Scene3DItemEnum.object_mapping(layout_relation)
    obj_prompt = f'{layout_relation["task_desc"]} {object_mapping}'

    objs_desc = LAYOUT_DESCRIBER(obj_prompt, params=params)

    layout_info = LayoutInfo(layout_tree, layout_relation, objs_desc)

    visualizer = SceneTreeVisualizer(layout_info)
    os.makedirs(args.save_root, exist_ok=True)
    scene_graph_path = f"{args.save_root}/scene_tree.jpg"
    visualizer.render(save_path=scene_graph_path)
    with open(f"{args.save_root}/layout.json", "w") as f:
        json.dump(layout_info.to_dict(), f, indent=4)

    print(f"Scene hierarchy tree saved to {scene_graph_path}")
    print(f"Disassembled Layout: {layout_relation}")
    print(f"Layout Graph: {layout_tree}")
    print(f"Layout Descriptions: {objs_desc}")
