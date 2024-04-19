# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
