# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
History module
==============

Use this module to manage a history object.
"""

import tensorflow as tf

from typing import Any, Optional, Union, Tuple

from SNN2.src.decorators.decorators import c_logger


@c_logger
class History:

    def __init__(self, default: Optional[tf.Tensor] = None):
        self.history: Union[None, tf.Tensor] = default

    def update(self, obj: tf.Tensor) -> None:
        if self.history is None:
            self.history = obj
        else:
            self.history = tf.concat([self.history, obj], 0)

    def reset(self) -> None:
        self.history = None

    @property
    def shape(self) -> Union[None, tf.TensorShape]:
        if self.history is None:
            return None
        return self.history.shape
