# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pandas.compat import F
import tensorflow as tf
from tensorflow._api.v2.random import shuffle
from SNN2.src.model.callbacks.callbacksWrapper import Callback_Selector
from SNN2.src.model.history.history import History

from tensorflow.keras import optimizers
from tensorflow.keras import Model

from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH

from SNN2.src.decorators.decorators import RLModelHandlerWrapper
from SNN2.src.model.reinforcement.reinforcement import RLModelHandler

@RLModelHandlerWrapper
class ACModelHandler(RLModelHandler):
    """ReinforceModelHandler.

    Generate and manage a reinforcement model
	"""


    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.qoe_threshold = ast.literal_eval(self.params["qoe_threshold"])
        self.gamma = ast.literal_eval(self.params["gamma"])
        self.training_epochs = ast.literal_eval(self.params["training_epochs"])
        self.current_training_epoch = 0
        self.history.update({
                    "critic": History(),
                    "action_prob": History(),
                })

        self.model: Model = self.define_model(None, self.emb)
        self.write_msg("Reinforcement model initialized", level=LH.INFO)
        self.load()

        # Compile the network
        self.model = self.compile(self.params["ReinforceCustomModel"],
                                  self.gamma,
                                  optimizer=optimizers.Adam(float(self.params["ReinforcementAdamStrength"])),
                                  run_eagerly=True,
                                  weighted_metrics=[])
        self.write_msg("Reinforcement model compiled")

    def calculate_reward(self, flags: tf.Tensor,
                         expected_classes: tf.Tensor,
                         expected_qoe: tf.Tensor,) -> Tuple[int, Tuple[int, ...], Optional[Dict[str, Any]]]:
        qoe_flags = tf.reshape(tf.where(expected_qoe > self.qoe_threshold, 0, 1), [-1])
        self.write_msg(f"QoE flags: {qoe_flags}, positives: {len(tf.where(qoe_flags == 0))}, negatives: {len(tf.where(qoe_flags == 1))}")
        correct_qoe, wrong_qoe = self.get_correct_wrong(flags, qoe_flags)
        correct_cls, wrong_cls = self.get_correct_wrong(flags, expected_classes)
        reward, confusion_matrix, extra_data = self.reward_function(correct_qoe, wrong_qoe)
        tf_reward = tf.expand_dims(tf.convert_to_tensor([float(reward)]), 0)
        self.history["reward"].update(tf_reward)

        self.write_msg(f"qoe, correct: {correct_qoe.shape} wrong: {wrong_qoe.shape}")
        self.write_msg(f"cls, correct: {correct_cls.shape} wrong: {wrong_cls.shape}")
        self.write_msg(f"Reward obtained: {reward}")
        self.write_msg(f"Extra data: {extra_data}")
        self.log_memory_state()
        return reward, confusion_matrix, extra_data

    def get_action(self, current_state: tf.Tensor) -> Tuple[int, List[float], float]:
        action_probs, critic_value = self.model(current_state)

        action = np.random.choice(self.actions, p=np.squeeze(action_probs))
        log_action = tf.math.log(action_probs[0, action])
        tf_action = tf.expand_dims(tf.convert_to_tensor([float(action)]), 0)

        self.history["state"].update(current_state)
        self.history["action_prob"].update(tf.expand_dims(log_action, 0))
        self.history["critic"].update(critic_value)
        self.history["action"].update(tf_action)

        self.write_msg(f"Action probs: {action_probs}")
        self.write_msg(f"Critic value: {critic_value}")
        self.write_msg(f"Action chosen: {action, action_probs[0, action]}")
        self.log_memory_state()
        return action, action_probs.numpy(), critic_value.numpy()

    def train_epoch(self, *args, **kwargs):
        self.write_msg("Executing a training cycle on the RL Network")
        if len(self.memory) == 0:
            self.write_msg("The memory is empty, no training is possible")
            return

        examples = self.memory.take()
        csv_logger = Callback_Selector("csvLogger", self.csv_output, append=True)
        self.model.fit(examples, *args, callbacks=[csv_logger], **kwargs)

    def execute_episode(self,
                        dump_memory: bool = True,
                        train: bool = True) -> None:
        if dump_memory:
            self.update_memory()
        if train:
            self.train_epoch(shuffle=False,
                             epochs=self.current_training_epoch+self.training_epochs,
                             initial_epoch=self.current_training_epoch)
            self.current_training_epoch += self.training_epochs

