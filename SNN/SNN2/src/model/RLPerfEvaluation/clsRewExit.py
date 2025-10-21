# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tensorflow as tf

from SNN2.src.decorators.decorators import RLPerfEval_cls, c_logger, reward

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

def convert(val: Any, input_type: Type, Transformer: Callable) -> Any:
    if isinstance(val, input_type):
        val = Transformer(val)
    return val

@RLPerfEval_cls
@c_logger
class AvgRewardThreshold:

    def __init__(self, *args,
                 reward_threshold:  Optional[Union[str, float]] = None,
                 steps_to_consider: Optional[Union[str, float]] = None,
                 episode_period: Optional[Union[str, int]] = None,
                 episode_threshold: Optional[Union[str, int]] = None,
                 console_io: bool = True,
                 console_io_interval: int = 10,
                 **kwargs) -> None:
        assert not reward_threshold is None
        assert not steps_to_consider is None
        assert not episode_threshold is None
        self.avg_threshold: float = convert(reward_threshold, str, float)
        self.steps: int = convert(steps_to_consider, str, int)
        self.ep_threshold: int = convert(episode_threshold, str, int)
        self.ep_period: int = convert(episode_period, str, int)
        self.console_io = console_io
        self.console_io_interval = console_io_interval
        self.avg_rewards_record = None

    def __call__(self, *args,
                 current_episode: Optional[int] = None,
                 reward_history: Optional[tf.Tensor] = None,
                 **kwargs) -> bool:

        if reward_history is None:
            raise Exception(f"A reward history must be provided")

        rewards: np.ndarray = reward_history.numpy().flatten()
        assert not current_episode is None
        self.write_msg("Avg Reward threshold evaluation function")

        if current_episode < self.ep_threshold:
            self.write_msg(f"Too early to evaluate {current_episode}, required {self.ep_threshold}")
            return False

        required_steps = self.steps
        if required_steps == -1:
            required_steps= len(rewards)

        self.write_msg(f"There are {len(rewards)} values to which compute the average")
        self.write_msg(f"Required steps: {required_steps}")

        if required_steps > len(rewards):
            self.write_msg(f"There are only {len(rewards)}, required {required_steps}")
            return False

        last_rew = rewards[-required_steps:]
        self.write_msg(f"last {required_steps} rewards: {last_rew}")
        avg_rew = np.mean(last_rew)

        if self.avg_rewards_record is None:
            self.avg_rewards_record = [avg_rew]
        else:
            self.avg_rewards_record.append(avg_rew)

        if len(self.avg_rewards_record) > self.ep_period:
            self.avg_rewards_record = self.avg_rewards_record[-self.ep_period:]

        self.write_msg(f"Updated avg rewards record: {self.avg_rewards_record}")

        overall_avg = np.mean(self.avg_rewards_record)
        self.write_msg(f"The overall mean is {overall_avg}")

        if len(self.avg_rewards_record) < self.ep_period:
            self.write_msg(f"It's too early, it's necessary to accumulate at least {self.ep_threshold} rewards")
            return False

        if self.console_io and (current_episode % self.console_io_interval == 0):
            print(f"Episode {current_episode} - AVG reward: {overall_avg}, All rewards: {self.avg_rewards_record}")

        if overall_avg >= self.avg_threshold:
            print(f"The avg reward {overall_avg} is OVER the threshold {self.avg_threshold} Episode: {current_episode}")
            print(f"Current rewards: {self.avg_rewards_record}")
            self.write_msg(f"The avg reward {overall_avg} is OVER the threshold {self.avg_threshold} Episode: {current_episode}")
            return True

        self.write_msg(f"The avg reward {overall_avg} is BELOW the threshold {self.avg_threshold}")
        return False

    def reset(self) -> None:
        self.avg_rewards_record = None
