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
import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import c_logger, ccb

from typing import Callable, Union, Dict


@ccb
@c_logger
class controlledCrossEntropy(Callback):

    def __init__(self,
                 categoricalMatrix: Dict[str, tf.Variable],
                 d_val: tf.data.Dataset,
                 full_validation: tf.data.Dataset,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10):
        self.d_val = d_val
        self.flags = full_validation
        self.threshold = ast.literal_eval(threshold) if isinstance(threshold, str) else threshold
        self.batch_flag = True if threshold_unit == 'batch' else False
        self.multiplier = 1.0

        self.matrix = list(categoricalMatrix.values())[0]
        self.current_cycle = 0

    def __get_predictions(self, *args, **kwargs) -> tf.Tensor:
        probs = self.model.predict(self.d_val, verbose=0)
        return tf.convert_to_tensor(probs)

    @tf.autograph.experimental.do_not_convert
    def __get_corr_matrix(self, lg_pred: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        C = None
        C_nan = None
        for i in range(self.flags.shape[1]):
            j_tensor = None
            for j in range(self.flags.shape[1]):
                j_pred = lg_pred[:, j]
                # self.write_msg(f"j_pred {j}: {j_pred}")
                i_oneH = self.flags[:, i]
                # self.write_msg(f"i_oneH {i}: {i_oneH}")
                upper = -1*tf.gather(j_pred, tf.where(i_oneH == 1)[:, 0])
                upper = tf.reduce_sum(upper)
                # self.write_msg(f"Upper: {upper}")
                lower = tf.reduce_sum(i_oneH)
                # self.write_msg(f"Lower: {lower}")
                corr = upper/lower
                self.write_msg(f"C_{i},{j} = {corr}")
                if upper == 0.0 and lower == 0.0:
                    j_tensor = tf.constant(np.nan, shape=[self.flags.shape[1]])
                    self.write_msg(j_tensor)
                    self.write_msg(j_tensor.shape)
                    if C_nan is None:
                        C_nan = [i]
                    else:
                        C_nan.append(i)
                    break

                if j_tensor is None:
                    j_tensor = tf.expand_dims(corr, 0)
                else:
                    j_tensor = tf.concat([j_tensor, tf.expand_dims(corr, 0)], 0)

            self.write_msg(f"Row {i} correlation: {j_tensor}")
            if C is None:
                C = tf.expand_dims(j_tensor, 0)
            else:
                C = tf.concat([C, tf.expand_dims(j_tensor, 0)], axis=0)

        self.write_msg(f"C_nan: {C_nan}")

        max_no_nan = tf.math.reduce_max(tf.where(tf.math.is_nan(C), 0.0, C))
        min_no_nan = tf.math.reduce_min(tf.where(tf.math.is_nan(C), 20.0, C))
        C = tf.where(tf.math.is_nan(C), max_no_nan, C)

        mask_assign_min = np.identity(C.shape[0])
        mask_assign_min = tf.convert_to_tensor(mask_assign_min)
        C = tf.where(mask_assign_min == 1, min_no_nan, C)

        self.write_msg(f"C row {C_nan[0]}: {C[C_nan[0], :]}")
        C = tf.map_fn(lambda x: (x-min_no_nan)/(max_no_nan - min_no_nan), C)
        self.write_msg(f"Normalized C: {C}")
        self.write_msg(f"C row {C_nan[0]}: {C[C_nan[0], :]}")
        return C

    def __get_empower_matrix(self, p: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        i = 1
        limit = 0.001

        self.write_msg(p[i,:])
        self.write_msg(self.flags[i,:])
        C_pre = tf.where(p < limit, 0, p)
        self.write_msg(C_pre[i,:])

        C_pre = tf.where(self.flags, 0, C_pre)
        self.write_msg(C_pre[i,:])
        C_pre = self.flags - C_pre

        for i in range(self.flags.shape[1]):
            j_tensor = None
            for j in range(self.flags.shape[1]):
                j_pred = C[:, j]
                i_oneH = self.flags[:, i]
                upper = tf.gather(j_pred, tf.where(i_oneH == 1)[:, 0])
                upper = tf.reduce_sum(upper)

                lower = tf.reduce_sum(i_oneH)
                corr = upper/lower
                self.write_msg(f"C_{i},{j} = {corr}")
                if upper == 0.0 and lower == 0.0:
                    j_tensor = tf.constant(np.nan, shape=[self.flags.shape[1]])
                    self.write_msg(j_tensor)
                    self.write_msg(j_tensor.shape)
                    if C_nan is None:
                        C_nan = [i]
                    else:
                        C_nan.append(i)
                    break

                if j_tensor is None:
                    j_tensor = tf.expand_dims(corr, 0)
                else:
                    j_tensor = tf.concat([j_tensor, tf.expand_dims(corr, 0)], 0)

            self.write_msg(f"Row {i} correlation: {j_tensor}")
            if C is None:
                C = tf.expand_dims(j_tensor, 0)
            else:
                C = tf.concat([C, tf.expand_dims(j_tensor, 0)], axis=0)

        return C


    def step(self, logs=None) -> None:
        limit = 1e-1

        predictions = self.__get_predictions()
        # epsilon_ = tf.constant(1e-7, predictions.dtype.base_dtype)
        # predictions = tf.clip_by_value(predictions, epsilon_, 1.0 - epsilon_)

        self.write_msg(f"Predictions obtained: {predictions}")
        # lg_pred = tf.math.log(predictions)

        cor_matrix = self.__get_empower_matrix(predictions)
        self.write_msg(f"Correlation matrix: {cor_matrix}")
        raise Exception

        cor_matrix = 1 - cor_matrix
        self.write_msg(f"Correlation matrix: {cor_matrix}")

        cor_matrix = tf.where(cor_matrix < limit, 0.0, cor_matrix)
        self.write_msg(f"Correlation matrix: {cor_matrix}")

        # self.matrix.assign(cor_matrix)
        self.matrix.assign(tf.linalg.matmul(self.matrix, cor_matrix))
        self.write_msg(f"New matrix: \n{self.matrix.numpy()}")
        raise Exception

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
        self.write_msg(f"Training concluded with matrix {self.matrix}")
