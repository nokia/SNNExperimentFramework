# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import Huber
from SNN2.src.decorators.decorators import action, loss

from typing import Callable

@loss
def HuberActorCritic(*args,
                     **kwargs) -> Callable:
    def loss_fn(data, reward):
        action_probs = data[:, 0]
        critic_value = data[:, 1]
        returns = tf.reshape(reward, [-1])

        diff = returns - critic_value
        actor_losses = (action_probs*-1)*diff
        critic_losses = tf.math.pow(diff, 2)
        critic_losses = tf.math.reduce_sum(critic_losses)
        actor_losses = tf.math.reduce_sum(actor_losses)
        return actor_losses + critic_losses
    return loss_fn

