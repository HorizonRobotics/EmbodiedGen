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

import logging

from embodied_gen.utils.gpt_clients import GPTclient

logger = logging.getLogger(__name__)

DEFAULT_RESOLVE_PROMPT = """You are matching a user's description to exactly one object in a 3D scene.

Scene instance list (each is an identifier, may contain IDs like "kitchen_cabinet_9197760", "banana_001"):
{instance_list}

User description (what they want to refer to, e.g. "黄色水果", "the yellow fruit", "柜子", "oven"):
"{user_spec}"

Rules:
1. Pick the ONE instance from the list that best matches the user's description (semantic match: e.g. "黄色水果" -> banana, "柜子" -> cabinet).
2. If no instance matches, reply with exactly: NONE
3. Otherwise reply with the EXACT instance name from the list, nothing else (no quotes, no explanation).

Your reply (one line, exact instance name or NONE):"""


def resolve_instance_with_llm(
    gpt_client: GPTclient,
    instance_names: list[str],
    user_spec: str,
    prompt_template: str | None = None,
) -> str | None:
    """Map a user description to a single scene instance name via LLM semantic matching.

    E.g. user says "yellow fruit" and the scene has "banana_001" -> returns "banana_001".
    Returns None when there is no match or the LLM replies NONE; the caller should
    prompt the user that the object does not exist and ask for re-entry.

    Args:
        gpt_client: GPT client instance, e.g. embodied_gen.utils.gpt_clients.GPT_CLIENT.
        instance_names: List of scene instance names from FloorplanManager.get_instance_names().
        user_spec: User input, e.g. "yellow fruit", "柜子", "the table".
        prompt_template: Optional custom prompt; placeholders {instance_list} and {user_spec}.

    Returns:
        The matched instance name (exactly one of instance_names), or None if no match.
    """
    if not user_spec or not instance_names:
        return None

    template = prompt_template or DEFAULT_RESOLVE_PROMPT
    instance_list_str = "\n".join(f"- {n}" for n in instance_names)
    prompt = template.format(
        instance_list=instance_list_str,
        user_spec=(user_spec or "").strip(),
    )

    try:
        response = gpt_client.query(text_prompt=prompt)
    except Exception as e:
        logger.warning("LLM `resolve_instance_with_llm` query failed: %s", e)
        return None

    if not response:
        return None

    first_line = response.strip().split("\n")[0].strip()
    if first_line.upper() == "NONE":
        return None
    candidate = first_line.strip('"\'')

    if not candidate:
        return None

    names_lower = {n.lower(): n for n in instance_names}
    candidate_lower = candidate.lower()

    if candidate in instance_names:
        return candidate

    if candidate_lower in names_lower:
        return names_lower[candidate_lower]

    matches = [n for n in instance_names if candidate_lower in n.lower()]
    if len(matches) == 1:
        return matches[0]

    logger.debug(
        "resolve_instance_with_llm: LLM reply %r did not match any of %s",
        first_line,
        instance_names[:5],
    )
    return None
