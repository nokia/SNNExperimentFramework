# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
CURE algorithm test module
==========================

Use this module to test the CURE algorithm module
"""

from typing import List
import numpy as np
import tensorflow as tf
import pytest

from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;

from SNN2.src.model.layers.CURELayer.algorithm import TfCURE, TfCluster
from SNN2.src.model.layers.CURELayer.algorithm import TfClusters

def tf_round(t, decimals=0):
    mul = tf.constant(10**decimals, dtype=t.dtype)
    return tf.round(t * mul)/mul

class TestTfCluster():

    def test_init(self) -> None:
        val = tf.convert_to_tensor(read_sample(FCPS_SAMPLES.SAMPLE_LSUN))
        tf_cluster = TfCluster(val, 0)
        tf.debugging.assert_equal(val, tf_cluster.points)

    def test_init_exp(self) -> None:
        val = tf.convert_to_tensor(read_sample(FCPS_SAMPLES.SAMPLE_LSUN))
        with pytest.raises(AssertionError):
            tf_cluster = TfCluster(tf.expand_dims(val, 0), 0)

    @pytest.mark.parametrize("counter",
                             [(tf.convert_to_tensor([[[3, 3]]])),
                              (tf.convert_to_tensor([[3, 4, 5]]))])
    def test_point_distances_exp(self, counter) -> None:
        points = tf.convert_to_tensor([[1, 2], [1, 2], [2, 4], [4, 8]])
        tf_cluster = TfCluster(points, 0)
        with pytest.raises(AssertionError):
            distances = tf_cluster.my_dst(counter)

    def test_point_distances(self) -> None:
        points = tf.convert_to_tensor([[1, 2], [1, 2], [2, 4], [4, 8]])
        counters = tf.convert_to_tensor([[3, 3]], dtype=tf.float32)
        exp_distances = tf.convert_to_tensor([[2.2360678, 2.2360678, 1.4142135, 5.0990195]])
        tf_cluster = TfCluster(points, 0)
        distances = tf_cluster.my_dst(counters)
        tf.debugging.assert_equal(
                    tf_round(exp_distances, decimals=4),
                    tf_round(distances, decimals=4))

    @pytest.mark.timeout(0.5)
    def test_point_dst_long(self) -> None:
        val = tf.convert_to_tensor(read_sample(FCPS_SAMPLES.SAMPLE_LSUN))
        counters = tf.convert_to_tensor(read_sample(FCPS_SAMPLES.SAMPLE_LSUN))
        tf_cluster = TfCluster(val, 0)
        distances = tf_cluster.my_dst(counters)
        assert distances.shape == (403, 403)

    def test_points_distances(self) -> None:
        points = tf.convert_to_tensor([[1, 2], [1, 2], [2, 4], [4, 8]])
        counters = tf.convert_to_tensor([[3, 3], [1, 2]], dtype=tf.float32)
        exp_distances = tf.convert_to_tensor([[2.2360678, 2.2360678, 1.4142135, 5.0990195],
                                              [0.0, 0.0, 2.2361, 6.7082]])
        tf_cluster = TfCluster(points, 0)
        distances = tf_cluster.my_dst(counters)
        tf.debugging.assert_equal(
                    tf_round(exp_distances, decimals=4),
                    tf_round(distances, decimals=4))

    def test_points_furthest(self) -> None:
        points = tf.convert_to_tensor([[1, 2], [1, 2], [2, 4], [4, 8]])
        counters = tf.convert_to_tensor([[3, 3], [1, 2]], dtype=tf.float32)
        exp_furt = tf.convert_to_tensor([4, 8], dtype=tf.float32)
        tf_cluster = TfCluster(points, 0)
        furthest, _ = tf_cluster.get_furthest(counters)
        tf.debugging.assert_equal(exp_furt, furthest)

    def test_mean(self) -> None:
        points = tf.convert_to_tensor([[1, 2], [1, 2], [2, 4], [4, 8]])
        mean = tf.convert_to_tensor([2.0, 4.0])
        tf_cluster = TfCluster(points, 0)
        tf.debugging.assert_equal(mean, tf_cluster.mean)

    def test_compute_representors(self):
        val = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [1, 3, 3],
                                    [6, 7, 8], [9, 10, 11]])
        # tf.map_fn(fn=lambda t: print(tf.type_spec_from_value(t)), elems=input_data)
        # assert False
        tf_clusters = TfCluster(val, 0)
        tf_clusters.compute_representors(n_representors=5)
        exp_rep = tf.convert_to_tensor([[ 9., 10., 11. ],
                                        [ 1.,  2.,  3. ],
                                        [ 4.,  5.,  6. ],
                                        [ 6.,  7.,  8. ],
                                        [ 1.,  3.,  3. ]])
        tf.debugging.assert_equal(tf_clusters.rep, exp_rep)

    def test_compute_representors_full(self):
        input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
        # Allocate three clusters.
        cure_instance = cure(input_data, 1)
        cure_instance.process()
        representors = cure_instance.get_representors()
        val = tf.convert_to_tensor(input_data)

        # tf.map_fn(fn=lambda t: print(tf.type_spec_from_value(t)), elems=input_data)
        # assert False
        tf_clusters = TfCluster(val, 0)
        tf_clusters.compute_representors(n_representors=5, compression_factor=0.5)
        tf_rep = tf.squeeze(tf.convert_to_tensor(representors))
        # print(tf.sort(tf_round(tf_rep, decimals=4), axis=1))
        tf.debugging.assert_equal(
                tf.sort(tf_round(tf.reshape(tf.convert_to_tensor(representors), [-1]), decimals=4)),
                tf.sort(tf.reshape(tf_round(tf_clusters.rep, decimals=4), [-1])))

    def test_cluster_to_cluster_distance(self) -> None:
        val = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [1, 3, 3],
                                    [6, 7, 8], [9, 10, 11]])
        tf_cluster1 = TfCluster(val, 0)
        tf_cluster1.compute_representors(n_representors=3, compression_factor=0.5)
        tf_cluster2 = TfCluster(val, 1)
        tf_cluster2.compute_representors(n_representors=3, compression_factor=0.5)
        cc_dst = tf_cluster1.my_dst(tf_cluster2)
        assert cc_dst.numpy() == 0.0

    def test_distances(self) -> None:
        input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
        # Allocate three clusters.
        cure_instance = cure(input_data, 3)
        cure_instance.process()
        clusters: List[List[int]] = cure_instance.get_clusters()
        lens = [len(x) for x in clusters]
        tf_clusters = [TfCluster(tf.convert_to_tensor([input_data[p] for p in c]), id) for id, c in enumerate(clusters)]

        for c in tf_clusters:
            c.compute_representors(n_representors=5, compression_factor=0.5)

        d0 = tf_clusters[0].my_dst(tf_clusters[1])
        d1 = tf_clusters[0].my_dst(tf_clusters[2])
        d2 = tf_clusters[1].my_dst(tf_clusters[2])
        result = tf.constant([1.312, 1.814, 1.688], dtype=tf.float32)
        dst = tf.stack([d0, d1, d2])
        tf.debugging.assert_equal(tf_round(dst, decimals=3), result)

class TestTfCURE():
    """TestTfCURE.

    Test class for the CURE algorithm implemented through TF
	"""

    def test_default(self) -> None:
        assert True

    def test_validate_arguments(self) -> None:
        input_data = tf.convert_to_tensor(np.empty((0, 0)))
        with pytest.raises(AssertionError):
            tf_cure_instance = TfCURE(input_data, 3)

        input_data = tf.convert_to_tensor(np.empty((0, 5)))
        with pytest.raises(AssertionError):
            tf_cure_instance = TfCURE(input_data, 3)

        input_data = tf.convert_to_tensor(np.empty((5, 5)))
        with pytest.raises(AssertionError):
            tf_cure_instance = TfCURE(input_data, 0)

        with pytest.raises(AssertionError):
            tf_cure_instance = TfCURE(input_data, 5, number_of_representatives=0)

        with pytest.raises(AssertionError):
            tf_cure_instance = TfCURE(input_data, 5, compression_factor=-0.1)

    def test_queue_creation(self) -> None:
        input_data = tf.convert_to_tensor(read_sample(FCPS_SAMPLES.SAMPLE_LSUN))
        tf_cure_instance = TfCURE(input_data, 3)
        tf_cure_instance.create_queue()
        assert False

    def test_clustering(self) -> None:
        input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
        # Allocate three clusters.
        cure_instance = cure(input_data, 3)
        cure_instance.process()
        clusters = cure_instance.get_clusters()

        input_data_tf = tf.convert_to_tensor(input_data)
        tf_cure_instance = TfCURE(input_data_tf, 3)
        tf_cure_instance.process()
        tf_clusters = tf_cure_instance.clusters
        assert tf_clusters is not None

        np.testing.assert_equal(clusters, tf_clusters.numpy())

