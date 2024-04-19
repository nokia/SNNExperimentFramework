# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from keras import Model
import numpy as np
import tensorflow as tf

from SNN2.src.decorators.decorators import RLActionPolicies, c_logger

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


@RLActionPolicies
@c_logger
class nnPolicy:

    def __init__(self, *args,
                 model: Optional[Model] = None,
                 register: Optional[Callable] = None,
                 **kwargs) -> None:
        assert not model is None
        self.model = model
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
                 **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[tf.Tensor, tf.Tensor]]:
        action_probs, critic_value = self.model(tf.expand_dims(observation, 0))
        self.write_msg(f"Exploration, got the following probs: {action_probs} with {critic_value} critic value")

        self.registration(register=self.register, current_step=current_step)
        return self.post_process(action_probs, critic_value)


@RLActionPolicies
@c_logger
class nnPolicyEpsilonGreedy:

    def __init__(self, *args,
                 model: Optional[Model] = None,
                 register: Optional[Callable] = None,
                 epsilon: float = 1.0,
                 min_epsilon: float = 0.0,
                 epsilon_steps: int = 10000,
                 np_rng: Optional[Any] = None,
                 num_action: int = 3,
                 **kwargs) -> None:
        assert not model is None
        self.model = model
        self.register = register
        self.minus_eps = (epsilon-min_epsilon)/epsilon_steps
        self.np_rng = np_rng if not np_rng is None else np.random.default_rng()
        self.epsilon = epsilon
        self.min_eps = min_epsilon
        self.num_actions = num_action

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
                 **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[tf.Tensor, tf.Tensor]]:

        if self.np_rng.binomial(1, self.epsilon):
            self.write_msg(f"Exploration with epsilon_value {self.epsilon}")
            action = self.np_rng.integers(low=0, high=self.num_actions)
            action_probs = np.array([[0.0]*self.num_actions])
            action_probs[0][action] = 1.0
            critic_value = np.array([[0.0]])
            self.epsilon = self.epsilon if self.epsilon <= self.min_eps else self.epsilon-self.minus_eps
            self.registration(register=self.register, current_step=current_step, result="Exploration")
            return self.post_process(action_probs, critic_value)

        action_probs, critic_value = self.model(tf.expand_dims(observation, 0))
        self.write_msg(f"Exploration, got the following probs: {action_probs} with {critic_value} critic value")

        self.registration(register=self.register, current_step=current_step)
        return self.post_process(action_probs, critic_value)
