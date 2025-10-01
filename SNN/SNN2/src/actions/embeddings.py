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
# Copyright (C) 2025 Mattia Milani <mattia.milani@nokia.com>

"""
Embeddings actions
==================

Actions that can be applied to load, manipulate and same embeddings.
"""

from typing import List, Optional, Tuple, Union, Any, Dict, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from SNN2.src.decorators.decorators import action, f_logger, timeit
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.pickleHandler import PickleHandler as PkH

@action
def load_embeddings(emb_path: str = None,
                    pkl: PkH = None,
                    **kwargs) -> tf.Tensor:
    if emb_path is None:
        raise ValueError("No embedding path provided")

    if pkl is None:
        raise ValueError("No PickleHandler provided")

    emb_l = pkl.load(emb_path, **kwargs)
    emb_tf = tf.convert_to_tensor(emb_l, dtype=tf.float32)
    emb_tf = tf.squeeze(emb_tf)
    return emb_tf

@action
def compute_centroids(emb_tf: tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(tf.reduce_mean(emb_tf, axis=0), axis=0)

@action
def load_samples(samples_path: str = None,
                 pkl: PkH = None,
                 **kwargs) -> List[tf.Tensor]:
    if samples_path is None:
        raise ValueError("No samples path provided")

    if pkl is None:
        raise ValueError("No PickleHandler provided")

    samples = pkl.load(samples_path, **kwargs)
    return samples
