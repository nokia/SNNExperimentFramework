# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import math
import numpy as np
import tensorflow as tf
from scipy import optimize
from SNN2.src.model.RLObservationPP.clsCmExp import my_tf_round

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import c_logger, ccb
from SNN2.src.io.pickleHandler import PickleHandler as PH

from typing import Callable, Optional, Tuple, Union, Dict


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

        # probs_rev = 1-probs
        probs_lg = tf.math.log(probs)
        self.write_msg(probs_lg.shape)

        g = tf.transpose(probs_lg)
        self.write_msg(g.shape)

        return g

    def __compute_c_matrix(self, g: tf.Tensor,
                           Y: tf.Tensor,
                           pred_classes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
            reduced_all = tf.where(tf.range(Y.shape[1]) == i, 1., reduced_all)
            self.write_msg(f"reduced: {reduced}")
            C = tf.expand_dims(reduced, 0) if C is None else tf.concat([C, tf.expand_dims(reduced, 0)], axis=0)
            C_all = tf.expand_dims(reduced_all, 0) if C_all is None else tf.concat([C_all, tf.expand_dims(reduced_all, 0)], axis=0)

        return C, C_all

    def __ensure_triangular(self, C: tf.Tensor) -> tf.Tensor:
        C_t = tf.transpose(C)
        # Create lower triangular matrix using TensorFlow
        n = C.shape[0]
        indices = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        lower_triangular = tf.cast(indices[0] >= indices[1], tf.float32)
        self.write_msg(lower_triangular)
        C = tf.where(lower_triangular==1, C_t, C)
        return C

    def __fast_matrix(self, p: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        self.write_msg(p[0, :])
        predicted_classes = tf.argmax(p, axis=1)
        self.write_msg(predicted_classes[0])
        g = self.__probs_pre_processing(p)

        Y = self.flags > 0

        C, C_all = self.__compute_c_matrix(g, Y,  predicted_classes)
        C = self.__ensure_triangular(C) if self.ensure_triangular else C
        # C_all = self.__ensure_triangular(C_all) if self.ensure_triangular else C_all

        return C, C_all


    def acce(self,
                     x,
                     target: tf.Tensor,
                     output: tf.Tensor,
                     categoricalMatrix: tf.Variable,
                     modifiable_indexes: tf.Tensor,
                     axis=-1,
                     epsilon=1e-7,
                     **kwargs) -> float:
        if categoricalMatrix is None:
            raise Exception(f"A categorical matrix must be provided")
        if modifiable_indexes is None:
            raise Exception(f"An index couple must be provided")
        assert target is not None
        assert output is not None

        new_matrix = tf.where(tf.equal(modifiable_indexes, 1.0), x, categoricalMatrix)

        target = tf.linalg.matmul(target, categoricalMatrix)
        epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1.0 - epsilon_)
        output = tf.math.log(output)
        y=tf.reduce_sum(target * output, axis)
        res=-tf.sigmoid(y)
        return tf.reduce_mean(res).numpy()

    def acce_noTrick(self,
                     x,
                     target: tf.Tensor,
                     output: tf.Tensor,
                     categoricalMatrix: tf.Variable,
                     modifiable_indexes: tf.Tensor,
                     axis=-1,
                     epsilon=1e-7,
                     **kwargs) -> float:
        if categoricalMatrix is None:
            raise Exception(f"A categorical matrix must be provided")
        if modifiable_indexes is None:
            raise Exception(f"An index couple must be provided")
        assert target is not None
        assert output is not None

        new_matrix = tf.where(tf.equal(modifiable_indexes, 1.0), x, categoricalMatrix)

        target = tf.linalg.matmul(target, categoricalMatrix)
        epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1.0 - epsilon_)
        output = tf.math.log(output)
        y=tf.reduce_sum(target * output, axis)
        res=-y
        return tf.reduce_mean(res).numpy()

    def get_loss_function(self,
                          c_i: tf.Tensor,
                          i: int,
                          predictions_i: tf.Tensor,
                          tmp_new_c_i: tf.Tensor,
                          minimize: bool = True) -> Callable:
        worst_index = tf.where(tf.equal(c_i, tf.reduce_min(c_i)), 1., 0.)
        self.write_msg(f"Worst over {c_i} -> j-{worst_index}")
        pre_worst = tf.zeros((i, self.matrix.shape[0]))
        post_worst = tf.zeros((self.matrix.shape[0]-i-1, self.matrix.shape[1]))
        self.write_msg(f"Pre worst: {pre_worst.shape}")
        self.write_msg(f"Post worst: {post_worst.shape}")
        worst_index = tf.concat([pre_worst, tf.expand_dims(worst_index,0), post_worst], axis=0)
        self.write_msg(f"worst_index: {worst_index.shape}")
        assert worst_index.shape == self.matrix.shape

        matrix_class = tf.concat([pre_worst, tf.ones((1,self.matrix.shape[1])), post_worst], axis=0)
        tmp_new_m_i = tf.concat([pre_worst, tf.expand_dims(tmp_new_c_i,0), post_worst], axis=0)
        tmp_matrix = tf.where(matrix_class == 1.0, tmp_new_m_i, self.matrix)
        self.write_msg(tmp_matrix.shape)

        flags_i = tf.gather(self.flags, tf.where(tf.equal(self.flags[:, i], 1.0))[:, 0])
        self.write_msg(f"Flags i: {flags_i}")
        def call_loss(x):
            min_mul = 1 if minimize else -1
            res = self.acce_noTrick(x,
                                    flags_i,
                                    predictions_i,
                                    tmp_matrix,
                                    worst_index)
            return min_mul*res

        return call_loss

    def __gold_value_class(self,
                           c_i: tf.Tensor,
                           i: int,
                           predictions_i: tf.Tensor,
                           minimize_threshold: float = 0.5,
                           n: int = 2) -> tf.Tensor:
        self.write_msg(c_i)
        self.write_msg(f"Required i: {i}")
        current_class_matrix = self.matrix[i, :]
        self.write_msg(f"Current matrix state: {current_class_matrix}")
        self.write_msg(f"Predictions on class-i: {predictions_i}")

        avg_pred = tf.reduce_mean(predictions_i[:, i])
        self.write_msg(f"Average predictions class-{i} probability: {avg_pred} > {minimize_threshold}: {avg_pred > minimize_threshold}")

        number_of_worst_indexes = tf.gather(c_i, tf.where(c_i < 1.0)[:, 0])
        self.write_msg(number_of_worst_indexes)
        number_of_worst_indexes = tf.gather(number_of_worst_indexes, tf.where(number_of_worst_indexes < 0.0)[:, 0])
        self.write_msg(number_of_worst_indexes)
        n = n if len(number_of_worst_indexes) >= n else len(number_of_worst_indexes)
        self.write_msg(f"n: {n}")

        new_c_i = tf.where(c_i == 1.0, 1.0, 0.0)

        if avg_pred > minimize_threshold:
            self.write_msg(f"I should find the value that minimizes the loss function")
            for k in range(n):
                self.write_msg(f"I should find the value that maximizes the loss function")
                loss_fn = self.get_loss_function(c_i, i, predictions_i, new_c_i, minimize=True)
                xmin = optimize.minimize_scalar(loss_fn, bounds=(-1,1), method="bounded").x
                self.write_msg(xmin)
                self.write_msg(loss_fn(xmin))
                new_c_i = tf.where(tf.equal(c_i, tf.reduce_min(c_i)), -xmin, new_c_i)
                self.write_msg(f"New c_i: {new_c_i}")
        else:
            self.write_msg(f"I should find the value that maximizes the loss function")
            for k in range(n):
                self.write_msg(f"I should find the value that maximizes the loss function")
                loss_fn = self.get_loss_function(c_i, i, predictions_i, new_c_i, minimize=False)
                xmin = optimize.minimize_scalar(loss_fn, bounds=(-1,1), method="bounded").x
                self.write_msg(xmin)
                self.write_msg(-loss_fn(xmin))
                new_c_i = tf.where(tf.equal(c_i, tf.reduce_min(c_i)), xmin, new_c_i)
                c_i = tf.where(tf.equal(c_i, tf.reduce_min(c_i)), 0.0, c_i)
                self.write_msg(f"New c_i: {new_c_i}")

        return new_c_i

    def __gold_search(self, c: tf.Tensor, predictions) -> tf.Tensor:
        C = None
        for i in range(c.shape[0]):
            p_i = tf.gather(predictions, tf.where(tf.equal(self.flags[:, i], 1.0))[:, 0])
            get_new_gold_value = self.__gold_value_class(c[i, :], i, p_i)
            self.write_msg(f"new gold value: {get_new_gold_value}")
            C = tf.expand_dims(get_new_gold_value, 0) if C is None else tf.concat([C, tf.expand_dims(get_new_gold_value, 0)], axis=0)
        return C

    def step(self, logs=None) -> None:

        predictions = self.__get_predictions()
        epsilon_ = tf.constant(self.epsilon, predictions.dtype.base_dtype)
        predictions = tf.clip_by_value(predictions, epsilon_, 1.0 - epsilon_)

        cor_matrix, full_matrix = self.__fast_matrix(predictions)

        # cor_matrix = self.__gold_search(full_matrix, predictions)

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
