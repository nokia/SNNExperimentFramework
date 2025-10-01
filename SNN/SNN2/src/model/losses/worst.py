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

