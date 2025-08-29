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


from embodied_gen.utils.monkey_patches import monkey_patch_maniskill

monkey_patch_maniskill()
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers import RecordEpisode
from tqdm import tqdm
import embodied_gen.envs.pick_embodiedgen
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.log import logger
from embodied_gen.utils.simulation import FrankaPandaGrasper


@dataclass
class ParallelSimConfig:
    """CLI parameters for Parallel Sapien simulation."""

    # Environment configuration
    layout_file: str
    """Path to the layout JSON file"""
    output_dir: str
    """Directory to save recorded videos"""
    gym_env_name: str = "PickEmbodiedGen-v1"
    """Name of the Gym environment to use"""
    num_envs: int = 4
    """Number of parallel environments"""
    render_mode: Literal["rgb_array", "hybrid"] = "hybrid"
    """Rendering mode: rgb_array or hybrid"""
    enable_shadow: bool = True
    """Whether to enable shadows in rendering"""
    control_mode: str = "pd_joint_pos"
    """Control mode for the agent"""

    # Recording configuration
    max_steps_per_video: int = 1000
    """Maximum steps to record per video"""
    save_trajectory: bool = False
    """Whether to save trajectory data"""

    # Simulation parameters
    seed: int = 0
    """Random seed for environment reset"""
    warmup_steps: int = 50
    """Number of warmup steps before action computation"""
    reach_target_only: bool = True
    """Whether to only reach target without full action"""


def entrypoint(**kwargs):
    if kwargs is None or len(kwargs) == 0:
        cfg = tyro.cli(ParallelSimConfig)
    else:
        cfg = ParallelSimConfig(**kwargs)

    env = gym.make(
        cfg.gym_env_name,
        num_envs=cfg.num_envs,
        render_mode=cfg.render_mode,
        enable_shadow=cfg.enable_shadow,
        layout_file=cfg.layout_file,
        control_mode=cfg.control_mode,
    )
    env = RecordEpisode(
        env,
        cfg.output_dir,
        max_steps_per_video=cfg.max_steps_per_video,
        save_trajectory=cfg.save_trajectory,
    )
    env.reset(seed=cfg.seed)

    default_action = env.unwrapped.agent.init_qpos[:, :8]
    for _ in tqdm(range(cfg.warmup_steps), desc="SIM Warmup"):
        # action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(default_action)

    grasper = FrankaPandaGrasper(
        env.unwrapped.agent,
        env.unwrapped.sim_config.control_freq,
    )

    layout_data = LayoutInfo.from_dict(json.load(open(cfg.layout_file, "r")))
    actions = defaultdict(list)
    # Plan Grasp reach pose for each manipulated object in each env.
    for env_idx in range(env.num_envs):
        actors = env.unwrapped.env_actors[f"env{env_idx}"]
        for node in layout_data.relation[
            Scene3DItemEnum.MANIPULATED_OBJS.value
        ]:
            action = grasper.compute_grasp_action(
                actor=actors[node]._objs[0],
                reach_target_only=True,
                env_idx=env_idx,
            )
            actions[node].append(action)

    # Excute the planned actions for each manipulated object in each env.
    for node in actions:
        max_env_steps = 0
        for env_idx in range(env.num_envs):
            if actions[node][env_idx] is None:
                continue
            max_env_steps = max(max_env_steps, len(actions[node][env_idx]))

        action_tensor = np.ones(
            (max_env_steps, env.num_envs, env.action_space.shape[-1])
        )
        action_tensor *= default_action[None, ...]
        for env_idx in range(env.num_envs):
            action = actions[node][env_idx]
            if action is None:
                continue
            action_tensor[: len(action), env_idx, :] = action

        for step in tqdm(range(max_env_steps), desc=f"Grasping: {node}"):
            action = torch.Tensor(action_tensor[step]).to(env.unwrapped.device)
            env.unwrapped.agent.set_action(action)
            obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    logger.info(f"Results saved in {cfg.output_dir}")


if __name__ == "__main__":
    entrypoint()
