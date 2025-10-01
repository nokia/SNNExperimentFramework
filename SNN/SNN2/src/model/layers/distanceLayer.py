#!/usr/bin/env python
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
# Copyright (C) 2021 Mattia Milani <mattia.milani@studenti.unitn.it>

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers

from SNN2.src.decorators.decorators import cclayer

@cclayer
class distance(layers.Layer):
    """DistanceLayer.
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, name="DistanceLayer", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, positive, anchor, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

@cclayer
class ContrastiveDistance(layers.Layer):

    def __init__(self, name="DistanceLayer", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x, y):
        sum_square = tf.reduce_sum(tf.square(x - y), axis=-1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

@cclayer
class mahalanobis_distance(layers.Layer):
    """Mahalanobis distance layer.
    This layer should be used in order to calculate distances using the mahalanobis
    formula.
    """

    def __init__(self,
                 positive_inv_s: tf.Tensor,
                 negative_inv_s: tf.Tensor,
                 name="MahalanobisDistanceLayer",
                 **kwargs):
        super().__init__(name=name, **kwargs)

    # def drop(self, x: tf.Tensor, indexes: tf.Tensor, axis=0) -> tf.tensor:
    #     x = x.numpy()
    #     idx = indexes.numpy()
    #     x = np.delete(x, idx, axis=axis)
    #     return tf.convert_to_tensor(x)

    def compute_inv_cov_matrix(self, data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass
        # data_mean = tf.math.reduce_mean(data)
        # data_std = tf.math.reduce_std(data)
        # zero_std = tf.where(data_std == 0.0)[:, 0]

        # if len(zero_std) > 0:
        #     data = self.drop(zero_std, axis=1)
        #     data_mean = out_drop(data_mean, zero_std)
        #     data_std = out_drop(data_std, zero_std)

        # data = tf.map_fn(lambda tmp: (tmp-data_mean)/data_std, data)
        # cov = tfp.stats.covariance(data)
        # inv_cov = tf.linalg.inv(cov, adjoint=False, name=None)

        # return inv_cov, zero_std


    def two_samples_mahalanobis(self, x: tf.Tensor, y: tf.Tensor, cov: tf.Tensor):
        pass
        # diff = tf.subtract(x, y)
        # diff_t = tf.transpose(diff)
        # left_part = tf.linalg.matmul(diff, cov)
        # mahal = tf.linalg.matmul(left_part, diff_t)
        # return tf.math.sqrt(tf.linalg.tensor_diag_part(mahal))

    def call(self, positive, anchor, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
        # p_cov_matrix, p_zero_std = self.compute_inv_cov_matrix(positive)
        # n_cov_matrix, n_zero_std = self.compute_inv_cov_matrix(negative)
        # ap_distance = self.two_samples_mahalanobis(anchor, positive, p_cov_matrix)
        # an_distance = self.two_samples_mahalanobis(anchor, negative, n_cov_matrix)
        # return (ap_distance, an_distance)
