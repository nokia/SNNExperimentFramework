# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import tensorflow as tf

from SNN2.src.decorators.decorators import loss_parameters, loss_param

from typing import Union

@loss_param
def margin(value: Union[float, str], *args, **kwargs):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tf.Variable(value)

@loss_param
def categoricalMatrix(classes: Union[int, str], *args, **kwargs):
    if isinstance(classes, str):
        value = ast.literal_eval(classes)
    return tf.Variable(tf.eye(classes))

def fn(param, *args, **kwargs):
    if param in loss_parameters:
        return loss_parameters[param](*args, **kwargs)
    else:
        raise ValueError(f"Loss \"{param}\" not available")

