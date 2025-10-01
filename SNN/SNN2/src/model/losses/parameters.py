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

