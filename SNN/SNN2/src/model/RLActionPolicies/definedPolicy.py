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

from SNN2.src.decorators.decorators import RLActionPolicies, c_logger

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from SNN2.src.io.logger import LogHandler

from SNN2.src.model.RLActionPolicies.nnPolicy import nnPolicy


@RLActionPolicies
@c_logger
class slowIncrease():

    def __init__(self, *args,
                 register: Optional[Callable] = None,
                 **kwargs) -> None:
        self.register = register

    def np_post_process(self, act_p: np.ndarray, c_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.squeeze(act_p), np.squeeze(c_v)

    def tf_post_process(self, act_p: np.ndarray, c_v: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        tf_action_probs = tf.expand_dims(tf.convert_to_tensor(act_p), 0)
        tf_action_probs = tf.cast(tf_action_probs, tf.float32)
        tf_critic_value = tf.expand_dims(tf.convert_to_tensor(c_v), 0)
        tf_critic_value = tf.cast(tf_critic_value, tf.float32)
        return tf_action_probs, tf_critic_value

    def post_process(self, act_p: np.ndarray, c_v: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[tf.Tensor, tf.Tensor]]:
        np_res = self.np_post_process(act_p, c_v)
        tf_res = self.tf_post_process(act_p, c_v)
        return np_res, tf_res

    def registration(self,
                     register: Optional[Callable] = None,
                     current_step: Optional[int] = None,
                     result: str = "Exploitation") -> None:
        if register is not None:
            register("binomialResult", result, step=current_step)


    def __call__(self,
                 observation: tf.Tensor,
                 *args,
                 current_step: Optional[int] = None,
                 interval: int = 10,
                 **kwargs) -> None:
        logger, write_msg = kwargs["logger"], kwargs["write_msg"]

        if current_step is None or not isinstance(current_step, int):
            raise Exception(f"Current step must be provided and be an integer, {current_step} was provided")

        action_probs, critic_value = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0]])
        if current_step % interval == 0:
            action_probs, critic_value = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0]])

        self.write_msg(f"Exploration, got the following probs: {action_probs} with {critic_value} critic value")
        self.registration(register=self.register, current_step=current_step)
        return self.post_process(action_probs, critic_value)
