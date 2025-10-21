# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf

from SNN2.src.decorators.decorators import loss

from typing import Union, Callable

@loss
def categoricalCrossentropy(*args,
                            axis=-1,
                            epsilon=1e-7,
                            **kwargs) -> Callable:
    def loss_fn(target, output):
        # scale preds so that the class probas of each sample sum to 1
        output = output / tf.reduce_sum(output, axis, True)
        # Compute cross entropy from probabilities.
        epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
        return -tf.reduce_sum(target * tf.math.log(output), axis)
    return loss_fn

def my_sigmoid(data,
               amplitude = 1.,
               smoothness = 2.,
               limit = 10) -> tf.Tensor:
    return limit*(amplitude/(amplitude+tf.math.exp(-(data/smoothness))))

@loss
def categoricalCrossentropyApple(*args,
                                 categoricalMatrix: Union[None, tf.Variable] = None,
                                 axis=-1,
                                 epsilon=1e-4,
                                 **kwargs) -> Callable:
    if categoricalMatrix is None:
        raise Exception(f"A categorical matrix must be provided")

    def loss_fn(target, output):
        target = tf.linalg.matmul(target, categoricalMatrix)
        epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1.0 - epsilon_)
        output = tf.math.log(output)
        y=tf.reduce_sum(target * output, axis)
        res=-tf.sigmoid(y)
        return res
    return loss_fn

@loss
def categoricalCrossentropyAppleNoSigmoid(*args,
                                          categoricalMatrix: Union[None, tf.Variable] = None,
                                          axis=-1,
                                          epsilon=1e-7,
                                          **kwargs) -> Callable:
    if categoricalMatrix is None:
        raise Exception(f"A categorical matrix must be provided")

    def loss_fn(target, output):
        target = tf.linalg.matmul(target, categoricalMatrix)
        epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1.0 - epsilon_)
        output = tf.where(output == epsilon, 1.0, output)
        output = tf.where((target == 1.0) & (output == 1.0), epsilon, output)
        output = tf.math.log(output)
        y=tf.reduce_sum(target * output, axis)
        res=-y
        return res
    return loss_fn

@loss
def acce_noTrick(*args,
                 categoricalMatrix: Union[None, tf.Variable] = None,
                 axis=-1,
                 epsilon=1e-7,
                 **kwargs) -> Callable:
    if categoricalMatrix is None:
        raise Exception(f"A categorical matrix must be provided")

    def loss_fn(target, output):
        epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
        target = tf.linalg.matmul(target, categoricalMatrix)
        output = tf.clip_by_value(output, epsilon, 1.0 - epsilon_)
        output = tf.math.log(output)
        y=tf.reduce_sum(target * output, axis)
        res=-y
        return res
    return loss_fn
