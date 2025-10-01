# !/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the graphNU grapheneral Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# graphNU grapheneral Public License for more details.
#
# You should have received a copy of the graphNU grapheneral Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2021 Mattia Milani <mattia.milani@nokia.com>

"""
metrics functions Wrapper
=========================

Use this module to load a custom metric function

"""

import tensorflow as tf

from tensorflow.keras.metrics import Metric
from SNN2.src.decorators.decorators import metric

from typing import Union

@metric
class distanceAccuracy(Metric):
    """distance_accuracy.
    Metrics that measure the accuracy of the distances
    """

    def __init__(self, name="distance_accuracy",
                 margin: Union[float, str] = 0.5,
                 **kwargs):
        """__init__.

        Parameters
        ----------
        name :
            name
        margin : float
            margin value used to determine if a prediction is correct or not
        kwargs :
            kwargs
        """
        if isinstance(margin, str):
            margin = float(margin)

        super().__init__(name=name, **kwargs)
        self.margin = tf.constant(margin)
        self.value_sum = self.add_weight(name="value_sum", initializer="zeros")
        self.samples_tot = self.add_weight(name="samples_tot",
                                           initializer="zeros",
                                           dtype="int32")

    def update_state(self, ap, an, sample_weight=None):
        """update_state.

        Parameters
        ----------
        ap :
            ap, anchor positive distances
        an :
            an, anchor negative distances
        sample_weight :
            sample_weight
        """
        accuracy = tf.greater(an, ap + self.margin)
        accuracy = tf.reduce_sum(tf.cast(accuracy, tf.float32))
        self.value_sum.assign_add(accuracy)

        num_samples = tf.shape(ap)[0]
        self.samples_tot.assign_add(num_samples)

    def result(self):
        """result.
        """
        res = self.value_sum/tf.cast(self.samples_tot, tf.float32)
        return res

    def reset_state(self):
        """reset_state.
        """
        self.value_sum.assign(0.)
        self.samples_tot.assign(0)
