#!/usr/bin/env python
# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
