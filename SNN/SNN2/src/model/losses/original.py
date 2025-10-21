# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf

from SNN2.src.decorators.decorators import loss

from typing import Union, Callable

@loss
def original(margin: Union[float, str, tf.Variable] = 0.5,
            **kwargs) -> Callable:
    if isinstance(margin, str):
        margin = tf.Variable(margin)

    def loss_fn(ap, an):
        diff = ap - an
        loss = tf.maximum(diff + margin, 0.)
        return loss
    return loss_fn

@loss
def original_RLimit(margin: Union[float, str, tf.Variable] = 0.5,
                    limit: Union[float, str, tf.Variable] = 2.0,
                    power: Union[float, str, tf.Variable] = -10.0,
                    **kwargs) -> Callable:
    if isinstance(margin, str):
        margin = tf.Variable(margin)
    if isinstance(limit, str):
        limit = tf.Variable(limit)
    if isinstance(power, str):
        power = tf.Variable(power)

    def loss_fn(ap, an):
        diff = ap - an
        m = margin
        if margin > limit:
            m = margin*power
        loss = tf.maximum(diff + m, 0.)
        return loss
    return loss_fn

