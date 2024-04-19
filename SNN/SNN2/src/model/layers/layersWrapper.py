# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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

from SNN2.src.decorators.decorators import layers, clayer


from typing import Union, List, Tuple

@clayer
def lstm(n_nodes, *args, **kwargs) -> LSTM:
    if isinstance(n_nodes, str):
        n_nodes = int(n_nodes)
    if isinstance(kwargs["input_shape"], str):
        kwargs["input_shape"] = ast.literal_eval(kwargs["input_shape"])
    return LSTM(n_nodes, *args, **kwargs)

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
