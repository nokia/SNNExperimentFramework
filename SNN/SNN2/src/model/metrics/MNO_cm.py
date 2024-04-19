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

from typing import Optional, Union

class generic_cm_object(Metric):

    def __init__(self,
                 expected_labels: Optional[tf.Tensor] = None,
                 name="generic_cm_object",
                 margin: Union[float, str] = 0.5,
                 normalize: bool = True,
                 **kwargs):
        if isinstance(margin, str):
            margin = float(margin)

        super().__init__(name=name, **kwargs)
        self.normalize = tf.Variable(normalize)
        self.active = tf.Variable(False)
        self.expected_labels = expected_labels
        self.margin = tf.constant(margin)
        self.value_sum = self.add_weight(name="value_sum", initializer="zeros")
        self.samples_tot = self.add_weight(name="samples_tot",
                                           initializer="zeros",
                                           dtype="int32")

    def get_object(self, ap, an) -> tf.Tensor:
        raise NotImplementedError("Must be implemented by subclasses")

    def update_state(self, ap, an, sample_weight=None):
        if self.expected_labels is None:
            return
        if self.active:
            obj = self.get_object(ap, an)
            self.value_sum.assign_add(obj)
            num_samples = tf.shape(ap)[0]
            self.samples_tot.assign_add(num_samples)

    def result(self):
        """result.
        """
        if self.normalize:
            return self.value_sum/tf.cast(self.samples_tot, tf.float32)
        return self.value_sum

    def reset_state(self):
        """reset_state.
        """
        self.value_sum.assign(0.)
        self.samples_tot.assign(0)

@metric
class TP_confusion_matrix(generic_cm_object):

    def __init__(self,
                 *args,
                 name="TP_confusion_matrix",
                 **kwargs):

        super().__init__(*args, name=name, **kwargs)

    def get_object(self, ap, an) -> tf.Tensor:
        p_indexes = tf.where(self.expected_labels==0)
        an_positives = tf.gather(an, p_indexes)
        ap_positives = tf.gather(ap, p_indexes)
        tp = tf.greater(an_positives, ap_positives + self.margin)
        tp = tf.reduce_sum(tf.cast(tp, tf.float32))
        return tp

@metric
class FP_confusion_matrix(generic_cm_object):

    def __init__(self,
                 *args,
                 name="FP_confusion_matrix",
                 **kwargs):

        super().__init__(*args, name=name, **kwargs)

    def get_object(self, ap, an) -> tf.Tensor:
        indexes = tf.where(self.expected_labels==1)
        an_i = tf.gather(an, indexes)
        ap_i = tf.gather(ap, indexes)
        obj = tf.greater(ap_i + self.margin, an_i)
        obj = tf.reduce_sum(tf.cast(obj, tf.float32))
        return obj

@metric
class TN_confusion_matrix(generic_cm_object):

    def __init__(self,
                 *args,
                 name="TN_confusion_matrix",
                 **kwargs):

        super().__init__(*args, name=name, **kwargs)

    def get_object(self, ap, an) -> tf.Tensor:
        indexes = tf.where(self.expected_labels==1)
        an_i = tf.gather(an, indexes)
        ap_i = tf.gather(ap, indexes)
        obj = tf.greater(an_i, ap_i + self.margin)
        obj = tf.reduce_sum(tf.cast(obj, tf.float32))
        return obj

@metric
class FN_confusion_matrix(generic_cm_object):

    def __init__(self,
                 *args,
                 name="FN_confusion_matrix",
                 **kwargs):

        super().__init__(*args, name=name, **kwargs)

    def get_object(self, ap, an) -> tf.Tensor:
        indexes = tf.where(self.expected_labels==0)
        an_i = tf.gather(an, indexes)
        ap_i = tf.gather(ap, indexes)
        obj = tf.greater(ap_i + self.margin, an_i)
        obj = tf.reduce_sum(tf.cast(obj, tf.float32))
        return obj

@metric
class U_confusion_matrix(generic_cm_object):

    def __init__(self,
                 *args,
                 name="U_confusion_matrix",
                 **kwargs):

        super().__init__(*args, name=name, **kwargs)

    def get_object(self, ap, an) -> tf.Tensor:
        obj = tf.greater(self.margin, tf.math.abs(ap - an))
        obj = tf.reduce_sum(tf.cast(obj, tf.float32))
        return obj
