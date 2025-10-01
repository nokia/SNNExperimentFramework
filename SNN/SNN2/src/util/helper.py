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
# Copyright (C) 2020 Mattia Milani <mattia.milani.ext@nokia.com>

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import History

from typing import Any, Dict, List, Optional

def merge_histories(h1: History, h2: History) -> History:
    return {key: h1.history[key] + h2.history[key] for key in h1.history.keys()}

def chain_histories(histories: List[History]) -> History:
    if len(histories) == 0:
        return None
    h = histories.pop(0)
    for next_h in histories:
        h.history = merge_histories(h, next_h)
    return h

def dst2tensor(data: tf.data.Dataset,
               *args,
               dtype = None,
               convert: Optional[Any] = None,
               **kwargs) -> tf.Tensor:
    if dtype is not None:
        np_array = np.array(list(data.as_numpy_iterator()), dtype=dtype)
    else:
        np_array = np.array(list(data.as_numpy_iterator()))

    if convert is not None:
        np_array = np_array.astype(convert)
    return tf.convert_to_tensor(np_array, *args, **kwargs)
