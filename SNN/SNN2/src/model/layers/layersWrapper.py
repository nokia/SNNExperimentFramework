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

import functools

import ast
import tensorflow as tf

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout

from SNN2.src.decorators.decorators import layers, clayer


from typing import Union, List, Tuple

@clayer
def lstm(n_nodes, *args, **kwargs) -> LSTM:
    if isinstance(n_nodes, str):
        n_nodes = int(n_nodes)
    if isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    if "dropout" in kwargs.keys() and isinstance(kwargs["dropout"], str):
        kwargs["dropout"] = float(kwargs["dropout"])
        assert 0 <= kwargs["dropout"] <= 1
    if "recurrent_dropout" in kwargs.keys() and isinstance(kwargs["recurrent_dropout"], str):
        kwargs["recurrent_dropout"] = float(kwargs["recurrent_dropout"])
        assert 0 <= kwargs["recurrent_dropout"] <= 1
    return LSTM(n_nodes, *args, **kwargs)

@clayer
def dropout(rate, *args, **kwargs) -> Dropout:
    if isinstance(rate, str):
        rate = float(rate)
    if isinstance(kwargs["noise_shape"], str):
        kwargs["noise_shape"] = ast.literal_eval(kwargs["noise_shape"])
    if isinstance(kwargs["seed"], str):
        kwargs["seed"] = int(kwargs["seed"])
    return Dropout(rate, *args, **kwargs)

@clayer
def flatten(*args, **kwargs) -> Flatten:
    return Flatten()

@clayer
def dense(n_nodes, *args, **kwargs) -> Dense:
    if isinstance(n_nodes, str):
        n_nodes = int(n_nodes)
    if "input_shape" in kwargs.keys() and (kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    return Dense(n_nodes, *args, **kwargs)

@clayer
def input(*args, **kwargs) -> Input:
    if "shape"in kwargs.keys() and isinstance(kwargs["shape"], str):
        kwargs["shape"] = ast.literal_eval(kwargs["shape"])
    if "input_shape"in kwargs.keys() and isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    return Input(*args, **kwargs)

@clayer
def conv2D(*args, **kwargs) -> Input:
    if "shape"in kwargs.keys() and isinstance(kwargs["shape"], str):
        print(kwargs["shape"])
        kwargs["shape"] = ast.literal_eval(kwargs["shape"])
    if "input_shape"in kwargs.keys() and isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    if "filters"in kwargs.keys() and isinstance(kwargs["filters"], str):
        kwargs["filters"] = ast.literal_eval(kwargs["filters"])
    if "kernel_size"in kwargs.keys() and isinstance(kwargs["kernel_size"], str):
        kwargs["kernel_size"] = ast.literal_eval(kwargs["kernel_size"])
    return Conv2D(*args, **kwargs)

@clayer
def maxPooling2D(*args, **kwargs) -> Input:
    if "shape"in kwargs.keys() and isinstance(kwargs["shape"], str):
        kwargs["shape"] = ast.literal_eval(kwargs["shape"])
    if "input_shape"in kwargs.keys() and isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    if "pool_size"in kwargs.keys() and isinstance(kwargs["pool_size"], str):
        kwargs["pool_size"] = ast.literal_eval(kwargs["pool_size"])
    return MaxPooling2D(*args, **kwargs)

@clayer
def normalization(*args, **kwargs) -> Input:
    if "shape"in kwargs.keys() and isinstance(kwargs["shape"], str):
        kwargs["shape"] = ast.literal_eval(kwargs["shape"])
    if "input_shape"in kwargs.keys() and isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    if "pool_size"in kwargs.keys() and isinstance(kwargs["pool_size"], str):
        kwargs["pool_size"] = ast.literal_eval(kwargs["pool_size"])
    return Normalization(*args, **kwargs)

@clayer
def normalizationLambda(mean: tf.Tensor, std: tf.Tensor, *args, **kwargs) -> Input:
    if "shape"in kwargs.keys() and isinstance(kwargs["shape"], str):
        kwargs["shape"] = ast.literal_eval(kwargs["shape"])
    if "input_shape"in kwargs.keys() and isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    if "pool_size"in kwargs.keys() and isinstance(kwargs["pool_size"], str):
        kwargs["pool_size"] = ast.literal_eval(kwargs["pool_size"])
    preprocess_layer = Lambda(
        lambda x: (x - mean) /std,
        *args, **kwargs
    )
    return preprocess_layer

def Layer_selector(obj, *args, **kwargs):
    if obj in layers.keys():
        return layers[obj](*args, **kwargs)
    else:
        raise Exception(f"{obj} layer not found")
