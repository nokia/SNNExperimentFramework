# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tensorflow as tf

from SNN2.src.decorators.decorators import RLPerfEval, f_logger, reward

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

def convert(val: Any, input_type: Type, Transformer: Callable) -> Any:
    if isinstance(val, input_type):
        val = Transformer(val)
    return val

@RLPerfEval
@f_logger
def rewardThreshold(*args,
                    current_episode: Optional[int] = None,
                    reward_history: Optional[tf.Tensor] = None,
                    reward_threshold:  Optional[Union[str, float]] = None,
                    steps_to_consider: Optional[Union[str, float]] = None,
                    episode_threshold: Optional[Union[str, int]] = None,
                    console_io: bool = True,
                    console_io_interval: int = 10,
                    **kwargs) -> bool:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    if reward_history is None:
        raise Exception(f"A reward history must be provided")
    assert not current_episode is None
    assert not reward_threshold is None
    assert not steps_to_consider is None
    assert not episode_threshold is None
    write_msg("Reward threshold evaluation function")

    threshold: float = convert(reward_threshold, str, float)
    steps: int = convert(steps_to_consider, str, int)
    ep_threshold: int = convert(episode_threshold, str, int)
    rewards: np.ndarray = reward_history.numpy().flatten()

    if current_episode < ep_threshold:
        write_msg(f"Too early to evaluate {current_episode}, required {ep_threshold}")
        return False

    if steps == -1:
        steps = len(rewards)

    if steps > len(rewards):
        write_msg(f"There are only {len(rewards)}, required {steps}")
        return False

    last_rew = rewards[-steps:]
    write_msg(f"last {steps} rewards: {last_rew}")
    avg_rew = np.mean(last_rew)

    if console_io and (current_episode % console_io_interval == 0):
        print(f"Episode {current_episode} - AVG reward: {avg_rew}, required {threshold} to terminate")

    if avg_rew >= threshold:
        print(f"The avg reward {avg_rew} is OVER the threshold {threshold} Episode: {current_episode}")
        write_msg(f"The avg reward {avg_rew} is OVER the threshold {threshold} Episode: {current_episode}")
        return True

    write_msg(f"The avg reward {avg_rew} is BELOW the threshold {threshold}")
    return False
