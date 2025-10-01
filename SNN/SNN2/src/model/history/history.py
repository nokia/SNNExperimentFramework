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
