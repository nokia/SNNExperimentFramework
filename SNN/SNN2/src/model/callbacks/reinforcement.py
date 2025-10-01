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

import ast
import numpy as np
import tensorflow as tf
from tensorflow._api.v2.math import confusion_matrix
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.util.dataManger import DataManager

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import ccb
from SNN2.src.core.gray.grayWrapper import validation_flag

from typing import Any, Optional, Union

@ccb
class reinforcement(Callback):

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
        self.logger(f"{self.__class__.__name__}", f"{msg}", level=level)


    def __init__(self,
                 model,
                 env,
                 env_evaluation_obj: tf.data.Dataset,
                 env_evaluation_expected_classes: tf.Tensor,
                 env_evaluation_expected_qoe: tf.Tensor,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 initial_training_epochs: Union[int, str] = 1,
                 logger: Optional[LH] = None,
                 csv_output: Optional[str] = None) -> None:
        self.RLmodelH = model
        self.env = env
        self.logger: Optional[LH] = logger
        self.csv_output: Optional[str] = csv_output
        self.df = DataManager(["Episode", "Step", "State", "Statistic", "Value"])

        if isinstance(threshold, str):
            threshold = ast.literal_eval(threshold)
        if isinstance(initial_training_epochs, str):
            self.initial_training_epochs = ast.literal_eval(initial_training_epochs)

        self.batch_flag = True if threshold_unit == 'batch' else False
        self.threshold: int = threshold
        self.initial_training_epochs: int = initial_training_epochs
        self.MAX_STEPS = self.RLmodelH.MAX_STEPS
        self.steps_executed = 0
        self.episdoes_executed = 0
        self.episode_reward = 0
        self.previous_state = None
        self.env_evaluation_obj = env_evaluation_obj
        self.env_evaluation_expected_classes = env_evaluation_expected_classes
        self.env_evaluation_expected_qoe = env_evaluation_expected_qoe
        self.waiting_for_reward = False

        self.RLmodelH.train_epoch(shuffle=False, epochs=self.initial_training_epochs, initial_epoch=0)

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

        action, action_probs, critic_value = self.RLmodelH.get_action(tf_state)

        self.previous_state = current_state
        self.write_msg(f"Execute action: {action}")
        self.env.execute_action(action)
        self.waiting_for_reward = True

        return (action, action_probs, critic_value)

    def episode_cycle(self):
        self.write_msg(f"---- required episdoe cycle {self.episdoes_executed+1} ----")
        self.RLmodelH.execute_episode()

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
        self.write_msg("---calculate step reward---")
        reward, confusion_matrix, extra_data = self.calculate_reward()
        self.waiting_for_reward = False
        self.write_msg(f"Reward obtained: {reward}")
        self.write_msg(f"Confusion matrix: {confusion_matrix}")
        self.write_msg(f"Extra data: {extra_data}")
        self.register("Reward", reward, step=self.steps_executed-1)
        self.register("TP", confusion_matrix[0], step=self.steps_executed-1)
        self.register("FP", confusion_matrix[1], step=self.steps_executed-1)
        self.register("TN", confusion_matrix[2], step=self.steps_executed-1)
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
        self.register("Action", action)
        self.register("ActionProbs", action_probs[0])
        self.register("CriticValue", critic_value[0][0])
        self.write_msg(f"Step cycle executed {action, action_probs, critic_value}")
        self.steps_executed += 1
        self.write_msg(f"current env state: {self.env.get_state()}")

    def cycle(self, counter):
        if (counter+1) % self.threshold != 0:
            return

        self.write_msg(f"current env state: {self.env.get_state()}")
        self.write_msg(f"Counter: {counter}")
        self.write_msg(f"Steps executed: {self.steps_executed}")

        if self.previous_state is not None:
            self.reward()

        self.write_msg(f"current env state: {self.env.get_state()}")
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

    def on_train_end(self, logs=None):
        if self.waiting_for_reward:
            self.write_msg(f"calculate last reward")
            self.reward()

        self.df.save(self.csv_output)
        self.episode()
        self.reset()
