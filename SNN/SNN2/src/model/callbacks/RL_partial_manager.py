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
from SNN2.src.decorators.decorators import c_logger, ccb
from SNN2.src.core.gray.grayWrapper import validation_flag

from typing import Any, Optional, Tuple, Union

@ccb
@c_logger
class RL_partial_env_manager(Callback):

    def __init__(self,
                 actor,
                 env,
                 d_val: tf.data.Dataset,
                 d_val_cls: tf.Tensor,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 initial_skip: Union[int, str] = 0,
                 exit_function: Optional[str] = None,
                 evaluation_margin: Union[float, str] = 0.5,
                 keep_fixed_margin: bool = False,
                 undecided_reverse: bool = False,
                 EnvExitFunctionH: Optional[Any] = None) -> None:
        self.RLmodelH = actor
        self.env = env

        self.threshold = ast.literal_eval(threshold) if isinstance(threshold, str) else threshold
        self.eval_margin = ast.literal_eval(evaluation_margin) if isinstance(evaluation_margin, str) else evaluation_margin

        self.batch_flag = True if threshold_unit == 'batch' else False
        self.steps_executed = 0
        self.episdoes_executed = 0
        self.d_val = d_val
        self.d_val_cls = d_val_cls
        self.last_cycle = False
        self.end = False
        self.training_interrupted = False
        self.EnvExitFunctionH = EnvExitFunctionH
        self.exit_function_name = "defaultEnvExit" if exit_function is None else exit_function
        self.exit_function = None
        self.keep_fixed_margin = keep_fixed_margin
        self.undecided_reverse = undecided_reverse
        if not self.EnvExitFunctionH is None:
            self.exit_function = self.EnvExitFunctionH.get_handler(self.exit_function_name)(
                        expected_labels=self.d_val_cls,
                        logger=self.logger)

    def __log_predictions(self, ap: tf.Tensor, an: tf.Tensor) -> None:
        self.write_msg(f"AP: {list(ap)}", level=LH.DEBUG)
        self.write_msg(f"AN: {list(an)}", level=LH.DEBUG)
        self.write_msg(f"AP > AN: {len(tf.where(ap > an))}", level=LH.DEBUG)
        self.write_msg(f"AP < AN: {len(tf.where(ap < an))}", level=LH.DEBUG)
        self.write_msg(f"AP == AN: {len(tf.where(ap == an))}", level=LH.DEBUG)

    def __log_labels(self, flags: tf.Tensor) -> None:
        self.write_msg(f"Flags 0: {len(tf.where(flags == 0))}", level=LH.DEBUG)
        self.write_msg(f"Flags 1: {len(tf.where(flags == 1))}", level=LH.DEBUG)
        self.write_msg(f"Flags 2: {len(tf.where(flags == 2))}", level=LH.DEBUG)

    def __get_predictions(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        ap, an = self.env.model.predict(self.d_val, verbose=0)
        self.__log_predictions(ap, an)
        return ap, an

    def __evaluate_dst(self, dst: Tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
        current_margin = round(self.env.state[0], 1) if not self.keep_fixed_margin else self.eval_margin
        self.write_msg(f"Model current margin settings: {current_margin}")
        self.write_msg(f"D_val expected labels: {list(self.d_val_cls.numpy())}", level=LH.DEBUG)
        flags: tf.Tensor = validation_flag(dst, self.d_val_cls,
                                           margin = current_margin,
                                           undecided_reverse=self.undecided_reverse)
        flags: tf.Tensor = tf.cast(flags, tf.int8)
        self.__log_labels(flags)
        return flags

    def get_labels(self) -> tf.Tensor:
        self.write_msg("Get labels", level=LH.DEBUG)
        dst: Tuple[np.ndarray, np.ndarray] = self.__get_predictions()
        labels: tf.Tensor = self.__evaluate_dst(dst)
        return labels

    def submit(self, action: int, state: np.ndarray) -> None:
        self.write_msg(f"Execute action: {action}")
        self.env.execute_action(action)
        self.write_msg(f"New environment state: {self.env.get_state()}")

    def step_cycle(self, logs=None) -> None:
        self.write_msg(f"---- Required step cycle {self.steps_executed+1} ----")
        self.write_msg(f"logs: {logs}")

        predictions: tf.Tensor = self.get_labels()
        current_state: np.ndarray = self.env.get_state()
        game_over: bool = self.end
        exit_flag = False

        if not self.exit_function is None:
            exit_flag = self.exit_function(predictions, current_state, self.steps_executed)
            self.write_msg(f"Exit flag: {exit_flag}")

        if not game_over and exit_flag and not self.training_interrupted:
            self.model.stop_training = True
            self.training_interrupted = True
            self.write_msg(f"Interrupting the training process")
            return

        self.write_msg(f"Current state: {current_state}")
        tf_state = tf.convert_to_tensor(current_state, dtype=tf.float32)
        action = self.RLmodelH.step(tf_state, predictions, game_over, cycle=self.steps_executed,
                                    interrupted=self.training_interrupted)

        if action is not None:
            self.submit(action, current_state)

    def step(self, logs=None) -> None:
        self.step_cycle(logs=logs)
        self.steps_executed += 1

    def cycle(self, counter, logs=None) -> None:
        if (counter+1) % self.threshold != 0:
            return

        self.write_msg(f"Current counter: {counter}")
        self.step(logs=logs)

    def reset(self) -> None:
        self.write_msg(f"Required reset")
        self.end = False
        self.last_cycle = False
        self.steps_executed = 0
        self.episdoes_executed += 1
        if not self.exit_function is None:
            self.exit_function.reset()

    def on_train_batch_end(self, batch, logs=None) -> None:
        if self.batch_flag:
            self.cycle(batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None) -> None:
        if not self.batch_flag:
            self.cycle(epoch, logs=logs)

    def on_train_end(self, logs=None) -> None:
        self.write_msg(f"The training is finished, signal the end of the episode to the actor")
        self.write_msg(f"Should I signal the game over? {self.last_cycle}")
        self.write_msg(f"has the training been interrupted? {self.training_interrupted}")
        if self.last_cycle or self.training_interrupted:
            self.write_msg(f"Current state of the RL-training value: {self.RLmodelH.trained}")
            self.end = True
            self.step(logs=logs)
            self.reset()
