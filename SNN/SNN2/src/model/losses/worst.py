# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf

from SNN2.src.decorators.decorators import loss

from typing import Any, Union, Callable

@loss
def wrong(margin: Union[float, str, tf.Variable] = 0.5,
            **kwargs) -> Callable:
    if isinstance(margin, str):
        margin = tf.Variable(margin)

    def loss_fn(ap, an) -> Any:
        diff = ap - an
        m = (2*an - 2*ap) + margin
        loss = tf.maximum(diff + m, 0.)
        return loss
    return loss_fn

