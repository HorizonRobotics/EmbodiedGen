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


from __future__ import annotations

import random

import json_repair
from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient
from embodied_gen.utils.log import logger

__all__ = [
    "InfinigenGenRouter",
    "DEFAULT_ROUTER_PROMPT",
]

DEFAULT_ROUTER_PROMPT = """
You are given a natural-language description of a household task or an indoor
scene involving objects.

Select all rooms from the predefined list below where this task or scene could
plausibly occur:
["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "DiningRoom", "Office"]

Rules:
1. Output must be a valid JSON nested array (2D list).
   Format: [["Room_List"], "Complexity_Level"]

2. Room Selection Logic (Index 0):
   - Standard Case: If the task is contained within specific rooms, select
     the relevant room name(s).
   - Special Case: If the task involves moving objects between different rooms
     (navigation/transport) or implies generating/referencing a complete house
     layout, use ["House"].
   - If no rooms are suitable and it is not a "House" case, randomly select
     one room.

3. Complexity Evaluation Logic (Index 1):
   - General Rule: Default to "medium".
   - Overrides (Apply these only if the description fits the specific
     criteria below):
     - "minimalist": If explicitly stated as minimalist.
     - "simple": If the scene is explicitly described as "simple", "basic".
     - "detail": If the scene is described as "complex", "detailed".

4. Do not include any explanations or additional text.

Example 1:
Task: minimalist/empty Bedroom.
Answer: [["Bedroom"], "minimalist"]

Example 2:
Task: Wiping the table in an simple room.
Answer: [["DiningRoom", "Kitchen"], "simple"]

Example 3:
Task: Take the vase from the living room shelf and navigate to the bedroom
to pack it.
Answer: [["House"], "medium"]

Example 4:
Task: Put the apple into the fruit bowl in a complex/detailed env.
Answer: [["Kitchen", "DiningRoom", "LivingRoom"], "detail"]

Task: {prompt}
Answer:
"""


class InfinigenGenRouter:
    """Router that maps task descriptions to room(s) and complexity via GPT."""

    def __init__(
        self,
        gpt_client: GPTclient,
        prompt: str | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            gpt_client: Client used to query the LLM.
            prompt: Optional custom system/template prompt. Uses
                DEFAULT_ROUTER_PROMPT if None.

        """
        self.gpt_client = gpt_client
        self.prompt = prompt if prompt is not None else DEFAULT_ROUTER_PROMPT

    def query(self, task_description: str) -> tuple[str, str]:
        """Map a task description to a room and complexity level.

        Args:
            task_description: Natural-language description of the task or scene.

        Returns:
            Tuple of (room_name, complexity_level). room_name is one room
            chosen at random from the list of rooms returned by the LLM.

        """
        filled_prompt = self.prompt.format(prompt=task_description)
        response_text = self.gpt_client.query(text_prompt=filled_prompt)
        if response_text is None:
            raise RuntimeError(
                "GPT router got no response. Check the GPT agent setup in "
                "embodied_gen/utils/gpt_config.yaml and network access."
            )
        parsed = json_repair.loads(response_text)

        room_list = parsed[0]
        complexity = parsed[1]
        room_name = random.choice(room_list)

        return room_name, complexity


def main() -> None:
    """Demo: run the router on a sample task."""
    agent = InfinigenGenRouter(gpt_client=GPT_CLIENT)
    room, complexity = agent.query(
        "Put the apple into the fruit bowl, complex env"
    )
    logger.info(f"Room: {room}, Complexity: {complexity}.")


if __name__ == "__main__":
    main()
