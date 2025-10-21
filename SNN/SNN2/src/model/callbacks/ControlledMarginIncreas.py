# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import math
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import c_logger, ccb

from typing import Callable, Union, Dict

def sigmoid(k: float = 3.,
            adjusting_value: float = 15.,
            limit: float = 10,
            precision: int = 4) -> Callable:
    """apply_sigmoid_fn.
    Function used to apply a sigmoid to the input

    Parameters
    ----------
    k : float
        k parameter used to adjust the curvature of the sigmoid
    adjusting_value : float
        adjusting_value parameter used to adjust the center of
        the sigmoid

    Returns
    -------
    Callable, the sigmoid function configured as required with a
    precision of 4 decimal points

    """
    def sigmoid(x: float):
        value = limit/(1 + math.pow(math.e,-((x-adjusting_value)/k)))
        value = round(value, precision)
        return value
    return sigmoid


@ccb
@c_logger
class controlledMargin(Callback):

    def __init__(self,
                 margin: Dict[str, tf.Variable],
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 delta: Union[float, str] = 10.0):
        self.threshold = ast.literal_eval(threshold) if isinstance(threshold, str) else threshold
        self.batch_flag = True if threshold_unit == 'batch' else False
        self.sigmoid = sigmoid()
        self.minimum = 0.5
        self.maximum = 10.0

        if isinstance(delta, str):
            delta = ast.literal_eval(delta)
        self.delta = delta
        self.margin = list(margin.values())[0]
        self.current_cycle = 0

    def step(self, logs=None) -> None:
        sig_value = self.sigmoid(self.current_cycle)
        if sig_value < self.minimum:
            sig_value = self.minimum
        if sig_value > self.maximum:
            sig_value = self.maximum
        self.margin.assign(sig_value)
        self.write_msg(f"Current margin value: {self.margin}")

    def cycle(self, counter, logs=None) -> None:
        if (counter+1) % self.threshold != 0:
            return

        self.write_msg(f"Current counter: {counter}")
        self.step(logs=logs)
        self.current_cycle += 1

    def on_train_batch_end(self, batch, logs=None) -> None:
        if self.batch_flag:
            self.cycle(batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None) -> None:
        if not self.batch_flag:
            self.cycle(epoch, logs=logs)

    def on_train_end(self, logs=None) -> None:
        self.write_msg(f"Training concluded with margin {self.margin}")
        return
