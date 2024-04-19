# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class module
============

Use this module to manage a class object.
"""

import numpy as np
import tensorflow as tf
from typing import Callable, List, Optional, Tuple, Type, Union

from SNN2.src.contextManagers.contextManagers import timeit_cnt
from SNN2.src.decorators.decorators import timeit
from SNN2.src.model.layers.CURELayer.tf_cluster import TfCluster, TfClusters
from scipy.spatial import KDTree
from rtree import index

class TfCURE:
    """TfCURE.

    This class is the entry point to execute the CURE algorithm using tensorflow
	"""


    def __init__(self, points_set: tf.Tensor,
                 number_of_clusters: int,
                 number_of_representatives: int = 5,
                 compression_factor: float = 0.5,
                 engine: str = "Default"):
        """__init__.
        Initialization method for the TfCURE object

        Parameters
        ----------
        points_set : tf.Tensor
            points_set
        number_of_clusters : int
            number_of_clusters
        number_of_representatives : int
            number_of_representatives
        compression_factor : float
            compression_factor
        """
        self.__clusters: Union[tf.Tensor, None] = None
        self.__queue: Union[TfClusters, None] = None
        self.__tree: Union[KDTree, None] = None
        self.__engine: str = engine

        self.__data_points: tf.Tensor = points_set
        self.__number_of_clusters: int = number_of_clusters
        self.__number_of_representatives: int = number_of_representatives
        self.__compression_factor: float = compression_factor

        self.__validate_arguments()

    def process(self) -> None:
        assert self.__engine == "Default", "The only engine supported up to now is 'Default'"
        self.__default_process()

    def __default_process(self) -> None:
        self.create_rtree()
        self.create_queue()
        self.iterate()

    def create_queue(self, data: Union[tf.Tensor, None] = None) -> None:
        data = self.__data_points if data is None else data
        # self.__queue = [cure_cluster(self.__pointer_data[index_point], index_point) for index_point in range(len(self.__pointer_data))]
        self.__queue = TfClusters(data,
                                  n_rep=self.__number_of_representatives,
                                  compression_factor=self.__compression_factor,
                                  tree=self.__tree)

    def create_kdtree(self, data: Union[tf.Tensor, None] = None) -> None:
        data = self.__data_points if data is None else data
        self.__tree = KDTree(data.numpy())

    def create_rtree(self, data: Union[tf.Tensor, None] = None) -> None:
        data = self.__data_points if data is None else data
        prop = index.Property(leaf_capacity=200, fill_factor=0.8)
        self.__tree = index.Index(properties=prop)

    def iterate(self) -> None:
        assert self.__queue is not None, "Must initialize the Queue before iteration"
        assert self.__tree is not None, "Must iterate the Tree before iteration"

        while self.__queue.size > self.__number_of_clusters:
            # The pop operation also removes u from the queue
            with timeit_cnt("Single iteration time"):
                with timeit_cnt("p1"):
                    u: TfCluster = self.__queue.pop()
                    v: TfCluster = self.__queue.get(u.closest)
                    self.__queue.remove(v)

                with timeit_cnt("p2"):
                    w: TfCluster = TfCluster.merge(u, v)
                    u.remove_from_tree()
                    v.remove_from_tree()
                    w.insert_into_tree()

                with timeit_cnt("p3"):
                    new_id = w.nearest_from_tree()
                    self.__queue.update_reverse_closest(new_id, w.id)
                    self.__recompute_distances(w, u, v)

                with timeit_cnt("p4"):
                    self.__queue.insert(w)
                    self.__queue.apply_sort()
            assert False

    def __recompute_distances(self, w: TfCluster,
                              u: TfCluster,
                              v: TfCluster) -> None:
        assert self.__queue is not None, "Must initialize the Queue before iteration"

        u.reverse_closests.remove(v.id)
        v.reverse_closests.remove(u.id)
        c_list = u.reverse_closests
        c_list.extend(v.reverse_closests)

        for x in self.__queue.iter:
            wx_dst = w.dst(x)

            # raise Exception
            if x.closest == u.id or x.closest == v.id:
                if x.distance_closest < wx_dst:
                    x.nearest_from_tree()
                else:
                    x.closest = (w.id, wx_dst)
            elif x.distance_closest > wx_dst:
                x.closest = (w.id, wx_dst)


    def cc(self, x: TfCluster,
           wdst: float) -> TfCluster:
        assert self.__tree is not None, "Must initialize the Tree before iteration"
        return self.__tree.closest_conditioned(x, limit=wdst)

    def __validate_arguments(self) -> None:
        assert not tf.equal(tf.size(self.__data_points), 0), f"The size of data points cannot be 0, shape: {self.__data_points.shape}, sie: {tf.size(self.__data_points)}"
        assert self.__number_of_clusters > 0, f"The number of clusters must be higher than 0, {self.__number_of_clusters} has been provided"
        assert self.__number_of_representatives > 0, f"The number of representatives must be higher than 0, {self.__number_of_representatives} has been provided"
        assert self.__compression_factor >= 0.0, f"The compression factor must be higher or equal to 0, {self.__compression_factor} has been provided"

    @property
    def clusters(self) -> Union[TfClusters, None]:
        return self.__clusters

    @property
    def tree(self) -> Union[KDTree, None]:
        return self.__tree
