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
import tensorflow_probability as tfp
from SNN2.src.model.RLObservationPP.clsCmExp import my_tf_round

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import c_logger, ccb
from SNN2.src.io.pickleHandler import PickleHandler as PH

from typing import Callable, Tuple, Union, Dict


@ccb
@c_logger
class controlledCrossEntropy(Callback):

    def __init__(self,
                 categoricalMatrix: Dict[str, tf.Variable],
                 d_val: tf.data.Dataset,
                 full_validation: tf.data.Dataset,
                 ph: PH,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 matrix_value: Union[int, str] = -1,
                 ensure_triangular: bool = True):
        self.d_val = d_val
        self.flags = full_validation
        self.threshold = ast.literal_eval(threshold) if isinstance(threshold, str) else threshold
        self.matrix_value = ast.literal_eval(matrix_value) if isinstance(matrix_value, str) else matrix_value
        self.batch_flag = True if threshold_unit == 'batch' else False
        self.epsilon = 1e-7
        self.ph = ph
        self.ensure_triangular = ensure_triangular

        self.matrix = list(categoricalMatrix.values())[0]
        self.current_cycle = 0

    def __get_predictions(self, *args, **kwargs) -> tf.Tensor:
        probs = self.model.predict(self.d_val, verbose=0)
        return tf.convert_to_tensor(probs)

    def __probs_pre_processing(self, p: tf.Tensor) -> tf.Tensor:
        probs = my_tf_round(p, decimals=2)
        epsilon_ = tf.constant(self.epsilon, probs.dtype.base_dtype)
        probs = tf.clip_by_value(probs, 0.0, 1.0 - epsilon_)

        probs_rev = 1-probs
        probs_lg = tf.math.log(probs_rev)
        self.write_msg(probs_lg.shape)

        g = tf.transpose(probs_lg)
        self.write_msg(g.shape)

        return g

    def __compute_c_matrix(self, g: tf.Tensor,
                           Y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        C = None
        C_all = None
        for i in range(Y.shape[1]):
            class_one_hot = Y[:, i]
            divider = tf.reduce_sum(tf.where(class_one_hot, 1., 0.))
            if divider == 0:
                reduced_all = tf.zeros(Y.shape[1])
                reduced = tf.zeros(Y.shape[1])
            else:
                elems = tf.boolean_mask(g, class_one_hot, axis=1)
                reduced_all = tf.reduce_sum(elems, axis=-1)
                reduced_all = reduced_all / divider

                min = tf.reduce_min(reduced_all)
                reduced = tf.where(reduced_all == min, self.matrix_value, 0.0)

            reduced = tf.where(tf.range(Y.shape[1]) == i, 1., reduced)
            reduced_all = tf.where(tf.range(Y.shape[1]) == i, 1., reduced)
            self.write_msg(f"reduced: {reduced}")
            C = tf.expand_dims(reduced, 0) if C is None else tf.concat([C, tf.expand_dims(reduced, 0)], axis=0)
            C_all = tf.expand_dims(reduced_all, 0) if C_all is None else tf.concat([C_all, tf.expand_dims(reduced_all, 0)], axis=0)

        return C, C_all

    def __ensure_triangular(self, C: tf.Tensor) -> tf.Tensor:
        C_t = tf.transpose(C)
        lower_triangular = tfp.math.fill_triangular([1]*sum(range(C.shape[0]+1)))
        self.write_msg(lower_triangular)
        C = tf.where(lower_triangular==1, C_t, C)
        return C

    def __fast_matrix(self, p: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        g = self.__probs_pre_processing(p)

        Y = self.flags > 0
        self.write_msg(Y)
        self.write_msg(Y.shape)

        C, C_all = self.__compute_c_matrix(g, Y)
        C = self.__ensure_triangular(C) if self.ensure_triangular else C
        C_all = self.__ensure_triangular(C_all) if self.ensure_triangular else C_all
        self.write_msg(C)
        self.write_msg(C[0:10,:10])

        return C, C_all

    def step(self, logs=None) -> None:

        predictions = self.__get_predictions()
        epsilon_ = tf.constant(self.epsilon, predictions.dtype.base_dtype)
        predictions = tf.clip_by_value(predictions, epsilon_, 1.0 - epsilon_)

        if ensure_triangular:
            cor_matrix, full_matrix = self.__fast_matrix_triangular(predictions)
        else:
            cor_matrix, full_matrix = self.__fast_matrix(predictions)

        self.ph.save(cor_matrix, fcorr_matrix_iteration_{self.current_cycle}")
        self.ph.save(full_matrix, f"corr_matrix_iteration_{self.current_cycle}")
        self.write_msg(f"Correlation matrix: {cor_matrix}")

        self.matrix.assign(cor_matrix)

        # self.matrix.assign(tf.linalg.matmul(self.matrix, cor_matrix))
        self.write_msg(f"New matrix: \n{self.matrix.numpy()}")

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
