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
