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
import time
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
class RL_env_manager(Callback):

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
                 actor,
                 env,
                 d_val: tf.data.Dataset,
                 d_val_cls: tf.Tensor,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 initial_skip: Union[int, str] = 0,
                 logger: Optional[LH] = None) -> None:
        self.RLmodelH = actor
        self.env = env
        self.logger: Optional[LH] = logger

        if isinstance(threshold, str):
            threshold = ast.literal_eval(threshold)

        self.batch_flag = True if threshold_unit == 'batch' else False
        self.threshold: int = threshold
        self.steps_executed = 0
        self.episdoes_executed = 0
        self.d_val = d_val
        self.d_val_cls = d_val_cls
        self.end = False

        # print(self.RLmodelH.summary())
        # self.RLmodelH.train()

    def __get_accuracy(self, ap, an, margin):
        correct = tf.reduce_sum(tf.cast(tf.math.greater(an, ap + margin), tf.float32))
        accuracy = correct/tf.cast(tf.shape(ap)[0], tf.float32)
        return accuracy

    def get_labels(self) -> tf.Tensor:
        self.write_msg("Get labels")
        start = time.time()
        ap, an = self.env.model.predict(self.d_val, verbose=0)
        end = time.time()
        self.write_msg(f"Model prediction execution time: {end - start}")
        current_margin = round(self.env.state[0], 1)
        self.write_msg(f"Model current margin settings: {current_margin}")
        flags: tf.Tensor = validation_flag((ap, an),
                                            self.d_val_cls,
                                            margin = current_margin)
        flags: tf.Tensor = tf.cast(flags, tf.int8)

        self.write_msg(f"Accuracy on predictions m={current_margin}: {self.__get_accuracy(ap, an, current_margin)}")
        self.write_msg(f"AP > AN: {len(tf.where(ap > an))}")
        self.write_msg(f"AP < AN: {len(tf.where(ap < an))}")
        self.write_msg(f"AP == AN: {len(tf.where(ap == an))}")
        self.write_msg(f"Flags 0: {len(tf.where(flags == 0))}")
        self.write_msg(f"Flags 1: {len(tf.where(flags == 1))}")
        self.write_msg(f"Flags 2: {len(tf.where(flags == 2))}")
        return flags

    def submit(self, action: int, state: np.ndarray) -> None:
        self.env.execute_action(action)
        self.write_msg(f"Execute action: {action}")
        self.write_msg(f"New environment state: {self.env.get_state()}")


    def step_cycle(self) -> None:
        self.write_msg(f"---- Required step cycle {self.steps_executed+1} ----")

        start = time.time()
        game_over: bool = self.end
        predictions: Optional[tf.Tensor] = self.get_labels()
        current_state: np.ndarray = self.env.get_state()
        # current_state = np.append(current_state, self.steps_executed)
        end = time.time()
        self.write_msg(f"Step cycle get information execution time: {end - start}")

        start = time.time()
        self.write_msg(f"Current state: {current_state}")
        tf_state = tf.convert_to_tensor(current_state, dtype=tf.float32)
        # tf_state = tf.expand_dims(tf_state, 0)
        if game_over:
            self.write_msg(f"Passing game over, end of episode")
        action = self.RLmodelH.step(tf_state, predictions, game_over)
        end = time.time()
        self.write_msg(f"Step cycle get action execution time: {end - start}")

        start = time.time()
        if action is not None:
            self.submit(action, current_state)
        end = time.time()
        self.write_msg(f"Step cycle submit action execution time: {end - start}")

    def step(self) -> None:
        start = time.time()
        self.step_cycle()
        self.steps_executed += 1
        end = time.time()
        self.write_msg(f"Step cycle execution time: {end - start}")

    def cycle(self, counter) -> None:
        if (counter+1) % self.threshold != 0:
            return

        self.write_msg(f"Current counter: {counter}")
        self.step()

    def reset(self) -> None:
        self.write_msg(f"Required reset")
        self.end = False
        self.steps_executed = 0
        self.episdoes_executed += 1

    def on_train_batch_end(self, batch, logs=None) -> None:
        if self.batch_flag:
            self.cycle(batch)

    def on_epoch_end(self, epoch, logs=None) -> None:
        if not self.batch_flag:
            self.cycle(epoch)

    def on_train_end(self, logs=None) -> None:
        self.write_msg(f"The training is finished, signal the end of the episode to the actor")
        self.end = True
        self.step()
        self.reset()
