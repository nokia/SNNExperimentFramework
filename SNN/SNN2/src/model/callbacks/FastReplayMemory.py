# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import ccb
from SNN2.src.core.gray.grayWrapper import validation_flag
from SNN2.src.util.dataManger import DataManager
from SNN2.src.io.logger import LogHandler as LH

from typing import Callable, Union, Optional, Any

@ccb
class FastReplayMemory(Callback):

    def write_msg(self, msg: str, level: int = LH.INFO) -> None:
        """__write_msg.
        write a message into the log file with a defined log level

        parameters
        ----------
        msg : str
            msg to print
        level : int
            level default info

        returns
        -------
        none

        """
        if self.logger is None:
            return
        self.logger(f"{self.__class__.__name__}: {msg}", level)

    def __init__(self,
                 model,
                 env,
                 numpyRNG,
                 env_evaluation_obj: tf.data.Dataset,
                 env_evaluation_expected_classes: tf.Tensor,
                 env_evaluation_expected_qoe: tf.Tensor,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 logger: Optional[LH] = None,
                 csv_output: Optional[str] = None):
        self.logger = logger
        self.RLmodelH = model
        self.env = env
        self.numpy_rng = numpyRNG
        self.csv_output: Optional[str] = csv_output
        self.df = DataManager(["Episode", "Step", "State", "Statistic", "Value"])

        if isinstance(threshold, str):
            threshold = ast.literal_eval(threshold)

        self.batch_flag = True if threshold_unit == 'batch' else False
        self.threshold: int = threshold
        self.MAX_STEPS = self.RLmodelH.MAX_STEPS
        self.steps_executed = 0
        self.episdoes_executed = 0
        self.episode_reward = 0
        self.previous_state = None
        self.env_evaluation_obj = env_evaluation_obj
        self.env_evaluation_expected_classes = env_evaluation_expected_classes
        self.env_evaluation_expected_qoe = env_evaluation_expected_qoe
        self.num_actions = self.RLmodelH.actions
        self.waiting_for_reward = False

    def get_random_action(self) -> int:
        action = self.numpy_rng.integers(low=0, high=self.num_actions)
        # action = 1
        self.write_msg(f"Action selected: {action}")
        return action

    def __get_accuracy(self, ap, an, margin):
        correct = tf.reduce_sum(tf.cast(tf.math.greater(an, ap + margin), tf.float32))
        accuracy = correct/tf.cast(tf.shape(ap)[0], tf.float32)
        return accuracy

    def calculate_reward(self):
        self.write_msg("Calculating reward")
        ap, an = self.env.model.predict(self.env_evaluation_obj, verbose=0)
        self.write_msg(f"Accuracy on predictions m=0.5: {self.__get_accuracy(ap, an, 0.5)}")
        self.write_msg(f"Accuracy on predictions m=0.0: {self.__get_accuracy(ap, an, 0.0)}")
        self.write_msg(f"AP > AN: {len(tf.where(ap > an))}")
        self.write_msg(f"AP < AN: {len(tf.where(ap < an))}")
        self.write_msg(f"AP == AN: {len(tf.where(ap == an))}")
        flags: tf.Tensor = validation_flag((ap, an),
                                self.env_evaluation_expected_classes,
                                margin = 0.5)
        flags = tf.cast(flags, tf.int32)
        self.write_msg(f"Flags 0: {len(tf.where(flags == 0))}")
        self.write_msg(f"Flags 1: {len(tf.where(flags == 1))}")
        self.write_msg(f"Flags 2: {len(tf.where(flags == 2))}")
        return self.RLmodelH.calculate_reward(flags,
                    self.env_evaluation_expected_classes,
                    self.env_evaluation_expected_qoe
               )

    def step_cycle(self):
        self.write_msg(f"---- Required step cycle {self.steps_executed+1} ----")
        current_state = self.env.get_state()
        self.write_msg(f"Current state: {current_state}")
        tf_state = tf.convert_to_tensor(current_state)
        tf_state = tf.expand_dims(tf_state, 0)

        self.RLmodelH.history["state"].update(tf_state)
        action: int = self.get_random_action()
        tf_action = tf.expand_dims(tf.convert_to_tensor([float(action)]), 0)
        self.RLmodelH.history["action"].update(tf_action)

        self.previous_state = current_state
        self.write_msg(f"Execute action: {action}")
        self.env.execute_action(action)
        self.waiting_for_reward = True
        return action, [1/self.num_actions]*self.num_actions, -1

    def episode_cycle(self):
        self.write_msg(f"---- required episode cycle {self.episdoes_executed+1} ----")
        self.RLmodelH.execute_episode(dump_memory=False, train=False)

    def register(self, stat: str, value: Any, step: Optional[int] = None) -> None:
        if step is None:
            step = self.steps_executed
        update = {"Episode": self.episdoes_executed,
                  "Step": step,
                  "State": self.previous_state,
                  "Statistic": stat,
                  "Value": value}
        self.write_msg(f"Update the df with {update}")
        self.df.append(update)

    def reward(self):
        self.write_msg("--- calculate step reward ---")
        reward, confusion_matrix, extra_data = self.calculate_reward()
        self.waiting_for_reward = False
        self.write_msg(f"Reward obtained: {reward}")
        self.write_msg(f"Confusion matrix: {confusion_matrix}")
        self.register("Reward", reward, step=self.steps_executed-1)
        self.register("TP", confusion_matrix[0], step=self.steps_executed-1)
        self.register("TN", confusion_matrix[1], step=self.steps_executed-1)
        self.register("FP", confusion_matrix[2], step=self.steps_executed-1)
        self.register("FN", confusion_matrix[3], step=self.steps_executed-1)
        if extra_data is not None:
            for key in extra_data:
                self.register(key, extra_data[key], step=self.steps_executed-1)

    def episode(self):
        self.steps_executed = 0
        self.episode_cycle()
        self.episdoes_executed += 1

    def step(self):
        self.write_msg("Require step cycle")
        action, action_probs, critic_value = self.step_cycle()
        self.write_msg(f"Step cycle executed {action, action_probs, critic_value}")
        self.register("Action", action)
        self.register("ActionProbs", action_probs)
        self.register("CriticValue", critic_value)
        self.steps_executed += 1
        self.write_msg(f"current env state: {self.env.get_state()}")


    def cycle(self, counter):
        if (counter+1) % self.threshold != 0:
            return

        if self.previous_state is not None:
            self.reward()

        if self.steps_executed == self.MAX_STEPS:
            self.episode()

        self.step()

    def reset(self) -> None:
        self.previous_state = None

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_flag:
            self.cycle(batch)

    def on_epoch_end(self, epoch, logs=None):
        if not self.batch_flag:
            self.cycle(epoch)
        self.df.save(self.csv_output)

    def on_train_end(self, logs=None):
        if self.waiting_for_reward:
            self.write_msg(f"calculate last reward")
            self.reward()

        self.df.save(self.csv_output)
        self.episode()
        self.reset()
