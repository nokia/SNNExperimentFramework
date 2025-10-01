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
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import ccb

from typing import Union, Dict

@ccb
class testParam(Callback):

    def __init__(self,
                 parameters: Dict[str, tf.Variable],
                 end_train_flag: Union[bool, str] = False,
                 epoch_threshold: Union[int, str] = 10,
                 delta: Union[float, str] = 10.0):
        self.parameters = parameters
        if isinstance(epoch_threshold, str):
            threshold = ast.literal_eval(epoch_threshold)
        if isinstance(delta, str):
            delta = ast.literal_eval(delta)
        if isinstance(end_train_flag, str):
            end_train_flag = ast.literal_eval(end_train_flag)

        self.threshold = threshold
        self.delta = delta
        self.end_train_flag = end_train_flag

    def update_params(self):
        for key, param in self.parameters.items():
            param.assign_add(self.delta)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.threshold == 0 and not self.end_train_flag:
            self.update_params()

    def on_train_end(self, logs=None):
        if self.end_train_flag:
            self.update_params()
