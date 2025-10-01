# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

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

