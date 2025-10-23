# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import math
import numpy as np
import tensorflow as tf
from SNN2.src.model.RLObservationPP.clsCmExp import my_tf_round

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import c_logger, ccb
from SNN2.src.io.pickleHandler import PickleHandler as PH

from typing import Callable, Tuple, Union, Dict


@ccb
@c_logger
class AdvancedControlledCrossEntropy(Callback):

    def __init__(self,
                 categoricalMatrix: Dict[str, tf.Variable],
                 d_val: tf.data.Dataset,
                 full_validation: tf.data.Dataset,
                 ph: PH,
                 threshold_unit: str = 'ep',
                 threshold: Union[int, str] = 10,
                 matrix_value: Union[int, str] = -1,
                 ensure_triangular: bool = True,
                 dario_norm: bool = False):
        self.d_val = d_val
        self.flags = full_validation
        self.threshold = ast.literal_eval(threshold) if isinstance(threshold, str) else threshold
        self.matrix_value = ast.literal_eval(matrix_value) if isinstance(matrix_value, str) else matrix_value
        self.batch_flag = True if threshold_unit == 'batch' else False
        self.epsilon = 1e-7
        self.ph = ph
        self.ensure_triangular = ensure_triangular
        self.current_multiplier = -1.0
        self.multiplier_delta = 0.03
        self.dario_norm = dario_norm

        self.matrix = list(categoricalMatrix.values())[0]
        self.current_cycle = 0

    def __get_predictions(self, *args, **kwargs) -> tf.Tensor:
        probs = self.model.predict(self.d_val, verbose=0)
        return tf.convert_to_tensor(probs)

    def __probs_pre_processing(self, p: tf.Tensor) -> tf.Tensor:
        probs = my_tf_round(p, decimals=2)
        epsilon_ = tf.constant(self.epsilon, probs.dtype.base_dtype)
        probs = tf.clip_by_value(probs, 0.0, 1.0 - epsilon_)

        # probs_rev = 1-probs
        probs_lg = tf.math.log(probs)
        self.write_msg(probs_lg.shape)

        g = tf.transpose(probs_lg)
        self.write_msg(g.shape)

        return g

    def __min_policy(self, row: tf.Tensor, i: int) -> Tuple[tf.Tensor, tf.Tensor]:
        reduced_all = tf.where(tf.range(row.shape[0]) == i, 1., row)
        min = tf.reduce_min(reduced_all)
        if min == 0:
            reduced = reduced_all
        else:
            self.write_msg(f"reduce_all2: {reduced_all}")
            reduced = tf.where(reduced_all == min, self.matrix_value, 0.0)
            self.write_msg(f"reduce_all3: {reduced_all}")

        return reduced, reduced_all

    def __second_min_policy(self, row: tf.Tensor, i: int) -> Tuple[tf.Tensor, tf.Tensor]:
        tmp_row = tf.where(tf.range(row.shape[0]) == i, 0., row)
        only_bad = tf.gather(tmp_row, tf.where(tmp_row != 0)[:, 0])
        self.write_msg(f"bad_elem: {only_bad}")
        if len(only_bad) > 0:
            min = tf.reduce_max(only_bad)
            reduced = tf.where(tf.equal(row, min), self.matrix_value, 0.0)
            reduced = tf.where(tf.equal(row, 1.0), 1.0, reduced)
        else:
            reduced = row
        self.write_msg(f"Original row: {row}")
        self.write_msg(f"Reduced row: {reduced}")
        return reduced, row

    def friendsToEnemy(self, row: tf.Tensor, i: int) -> Tuple[tf.Tensor, tf.Tensor]:
        reduced = tf.math.multiply(row, self.current_multiplier)
        reduced = tf.where(tf.range(row.shape[0]) == i, 1., reduced)
        reduced = tf.clip_by_value(reduced, 0.0, 1.0)
        row = tf.where(tf.range(row.shape[0]) == i, 1., row)

        return reduced, row

    def __dario_norm(self, row: tf.Tensor, i: int):
        lower = -1
        upper = 1.
        k = 3
        mult = upper - lower
        self.write_msg(f"Row: {row}")
        tmp_row = tf.where(tf.range(row.shape[0]) == i, 0., row)
        bad_idx = tf.where(tmp_row != 0)[:, 0]
        if len(bad_idx) > 0:
            self.write_msg(f"bad_idx: {bad_idx}")
            only_bad = tf.gather(tmp_row, bad_idx)
            only_bad = tf.sort(only_bad, direction='DESCENDING')
            only_bad = only_bad[:k] if len(only_bad) > k else only_bad
            self.write_msg(f"bad_elem: {only_bad}")
            self.write_msg(f"tmp_row {tmp_row}")
            min = tf.math.reduce_min(only_bad)
            max = tf.math.reduce_max(only_bad)
            tile_multiples = tf.concat([tf.ones(tf.shape(tf.shape(tmp_row)), dtype=tf.int32), tf.shape(only_bad)], axis=0)
            x_tile = tf.tile(tf.expand_dims(tmp_row, -1), tile_multiples)
            x_in_s = tf.reduce_any(tf.equal(x_tile, only_bad), -1)
            tmp_row = tf.where(x_in_s, tmp_row, 0.0)

            if min == max:
                tmp_row = tf.where(x_in_s, lower, tmp_row)
            else:
                tmp_row = tf.math.add(tf.math.multiply(upper, tf.math.divide(tf.math.subtract(tmp_row, min), tf.math.subtract(max, min))), lower)
            self.write_msg(f"tmp_row {tmp_row}")
            tmp_row = tf.where(row != 0.0, tmp_row, 0.0)
            tmp_row = tf.where(row == 1.0, 1.0, tmp_row)
            self.write_msg(f"bad_elem: {tmp_row}")
        else:
            tmp_row = row
            self.write_msg(f"Nothing to do")
            self.write_msg(f"bad_elem: {tmp_row}")

        return tmp_row, row

    def __apply_ploicy(self, row: tf.Tensor, i: int) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.dario_norm:
            return self.__dario_norm(row, i)
        return self.__second_min_policy(row, i)

    def __slow_compute_c_matrix(self, g: tf.Tensor,
                                Y: tf.Tensor,
                                pred_class: tf.Tensor):
        C = None
        C_all = None
        for i in range(Y.shape[1]):
            j_tensor = None
            for j in range(Y.shape[1]):
                self.write_msg(f"{i}-{j}")
                if i == j:
                    val = tf.Variable(1.0)
                else:
                    expected_j = Y[:, j]
                    obtained_i = tf.where(tf.equal(pred_class, i), True, False)
                    self.write_msg(expected_j)
                    self.write_msg(expected_j.shape)
                    self.write_msg(obtained_i)
                    self.write_msg(obtained_i.shape)
                    I = tf.where(tf.equal(expected_j, True) & tf.equal(obtained_i, True), 1.0, 0.0)
                    total_I = tf.reduce_sum(I)
                    if total_I == 0:
                        val = tf.Variable(0.0)
                    else:
                        self.write_msg("There is a match")
                        self.write_msg(I)
                        self.write_msg(tf.reduce_sum(I))
                        probabilities = tf.gather(g[:, j], tf.where(tf.equal(I, 1.0))[:, 0])
                        self.write_msg(probabilities)
                        self.write_msg(probabilities.shape)
                        val = tf.reduce_sum(probabilities)
                        val = val/total_I

                self.write_msg(val.shape)
                self.write_msg(val)
                j_tensor = tf.expand_dims(val, 0) if j_tensor is None else tf.concat([j_tensor, tf.expand_dims(val, 0)], axis=0)
                self.write_msg(j_tensor)
                self.write_msg(j_tensor.shape)

            j_tensor_red, j_tensor = self.__apply_ploicy(j_tensor, i)
            C = tf.expand_dims(j_tensor_red, 0) if C is None else tf.concat([C, tf.expand_dims(j_tensor_red, 0)], axis=0)
            C_all = tf.expand_dims(j_tensor, 0) if C_all is None else tf.concat([C_all, tf.expand_dims(j_tensor, 0)], axis=0)
        return C, C_all

    def __compute_c_matrix(self, g: tf.Tensor,
                           Y: tf.Tensor,
                           pred_class: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        C = None
        C_all = None
        for i in range(Y.shape[1]):
            class_one_hot = Y[:, i]
            divider = tf.reduce_sum(tf.where(class_one_hot, 1., 0.))
            self.write_msg(f"divider: {divider}")
            if divider == 0:
                reduced_all = tf.zeros(Y.shape[1])
                reduced_all = tf.where(tf.range(Y.shape[1]) == i, 1., reduced_all)
                reduced = tf.zeros(Y.shape[1])
                reduced = tf.where(tf.range(Y.shape[1]) == i, 1., reduced)
            else:
                elems = tf.boolean_mask(g, class_one_hot, axis=1)
                reduced_all = tf.reduce_sum(elems, axis=-1)
                self.write_msg(f"Sum: {reduced_all}")
                reduced_all = reduced_all / divider
                self.write_msg(f"reduce_all: {reduced_all}")

                self.write_msg(f"Current mult: {self.current_multiplier}")
                reduced, reduced_all = self.__apply_ploicy(reduced_all, i)
                self.write_msg(f"{i} - reduced_all: {reduced_all}")
                self.write_msg(f"{i} - reduced: {reduced}")

            C = tf.expand_dims(reduced, 0) if C is None else tf.concat([C, tf.expand_dims(reduced, 0)], axis=0)
            C_all = tf.expand_dims(reduced_all, 0) if C_all is None else tf.concat([C_all, tf.expand_dims(reduced_all, 0)], axis=0)
        self.current_multiplier += self.multiplier_delta

        return C, C_all

    def __ensure_triangular(self, C: tf.Tensor) -> tf.Tensor:
        C_t = tf.transpose(C)
        n = C.shape[0]
        indices = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        lower_triangular = tf.cast(indices[0] >= indices[1], tf.float32)
        C = tf.where(lower_triangular==1, C_t, C)
        return C

    def __fast_matrix(self, p: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        g = self.__probs_pre_processing(p)
        pred_class = tf.argmax(p, axis=1)

        Y = self.flags > 0
        self.write_msg(Y)
        self.write_msg(Y.shape)

        C, C_all = self.__slow_compute_c_matrix(p, Y, pred_class)
        self.write_msg(f"Pre triangularization {C[:10, :10]}")
        C = self.__ensure_triangular(C) if self.ensure_triangular else C
        self.write_msg(f"Post triangularization {C[:10, :10]}")
        # C_all = self.__ensure_triangular(C_all) if self.ensure_triangular else C_all

        return C, C_all

    def step(self, logs=None) -> None:

        predictions = self.__get_predictions()
        epsilon_ = tf.constant(self.epsilon, predictions.dtype.base_dtype)
        predictions = tf.clip_by_value(predictions, epsilon_, 1.0 - epsilon_)

        cor_matrix, full_matrix = self.__fast_matrix(predictions)
        self.write_msg(f"Correlation matrix: {cor_matrix}")

        self.ph.save(cor_matrix, f"phi_matrix_iteration_{self.current_cycle}")
        self.ph.save(full_matrix, f"corr_matrix_iteration_{self.current_cycle}")

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
