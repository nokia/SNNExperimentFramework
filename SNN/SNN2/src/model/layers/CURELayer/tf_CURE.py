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
                 n_clusters: int,
                 number_of_representatives: int = 5,
                 compression_factor: float = 0.5,
                 engine: str = "Default"):
        """__init__.
        Initialization method for the TfCURE object

        Parameters
        ----------
        points_set : tf.Tensor
            points_set
        n_clusters : int
            n_clusters
        number_of_representatives : int
            number_of_representatives
        compression_factor : float
            compression_factor
        """
        self.__clusters: Union[tf.Tensor, None] = None
        self.__queue: Union[TfClusters, None] = None
        self.__tree: Union[KDTree, None] = None
        self.__engine: str = engine
        self.__dimension: int = points_set.shape[-1]

        self.__data_points: tf.Tensor = points_set
        self.__number_of_clusters: int = n_clusters
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
        prop = index.Property(leaf_capacity=200,
                              fill_factor=0.8,
                              dimension=self.dimension)
        self.__tree = index.Index(properties=prop)

    def rm_cluster_tree(self, cluster: TfCluster) -> None:
        """rm_cluster_tree.

        Remove a cluster from the local tree

        Parameters
        ----------
        cluster : TfCluster
            The cluster to remove

        Returns
        -------
        """
        for rep in cluster.rep.numpy():
            self.__tree.delete(cluster.id, rep)

    def iterate(self) -> None:
        assert self.__queue is not None, "Must initialize the Queue before iteration"
        assert self.__tree is not None, "Must iterate the Tree before iteration"

        while self.__queue.size > self.__number_of_clusters:
            # The pop operation also removes u from the queue
            with timeit_cnt(f"Queue size: {self.__queue.size}/{self.__number_of_clusters} Iteration time"):
                u: TfCluster = self.__queue.pop()
                print(f"U: {u.min_str()}")
                v: TfCluster = self.__queue.get(u.closest)
                print(f"V: {v.min_str()}")
                self.__queue.remove(v)

                w: TfCluster = TfCluster.merge(u, v)
                print(f"W: {w.min_str()}")
                print(f"tree analysis")
                print(f"Self tree id: {id(self.__tree)}")
                print(f"U tree id: {id(u.tree)}")
                print(f"V tree id: {id(v.tree)}")
                print(f"W tree id: {id(w.tree)}")
                u.remove_from_tree()
                self.rm_cluster_tree(u)
                v.remove_from_tree()
                self.rm_cluster_tree(v)
                if w.id == 117:
                    t = 6
                    c = list(self.__tree.nearest(w.rep.numpy()[0], t, objects="raw"))
                    for tmp in c:
                        print(f"Tmp: {tmp}")
                    raise Exception("Stop")

                w.insert_into_tree()

                if w.id == 117:
                    t = 6
                    c = list(self.__tree.nearest(w.rep.numpy()[0], t, objects="raw"))
                    for tmp in c:
                        print(f"Tmp: {tmp}")
                    raise Exception("Stop")
                    new_id = w.nearest_from_tree(debug_flag=True)
                else:
                    new_id = w.nearest_from_tree()
                print(f"New closest cluster {new_id}")
                self.__queue.update_reverse_closest(new_id, w.id)
                print(f"{new_id} cluster state: {self.__queue.get(new_id).min_str()}")
                self.__recompute_distances(w, u, v)
                print(f"{new_id} cluster state: {self.__queue.get(new_id).min_str()}")
                print(f"W: {w.min_str()}")

                self.__queue.insert(w)
                self.__queue.apply_sort()

    def __recompute_distances(self, w: TfCluster,
                              u: TfCluster,
                              v: TfCluster) -> None:
        assert self.__queue is not None, "Must initialize the Queue before iteration"

        w_lookup=4
        x_lookup=117
        if w.id == w_lookup:
            print(f"Recompute distances of {w_lookup}")
            print(f"U: {u.min_str()}")
            print(f"V: {v.min_str()}")
            print(f"W: {w.min_str()}")

        u.reverse_closests.remove(v.id)
        v.reverse_closests.remove(u.id)
        c_list = u.reverse_closests
        c_list.extend(v.reverse_closests)

        if w.id == w_lookup:
            print(f"Closests: {c_list}")

        for x in self.__queue.iter:
            if w.id == w_lookup and x.id == x_lookup:
                print("Comparing W-X")
                print(f"X: {x.min_str()}")

            wx_dst = w.dst(x)

            if w.id == w_lookup and x.id == x_lookup:
                print(f"Distance: {wx_dst}")

            if x.closest == u.id or x.closest == v.id:
                # This closest is not valid anymore but the closest needs to be
                # recomputed only if the new distance w-x is worst than before
                if x.distance_closest >= wx_dst:
                    if w.id == w_lookup and x.id == x_lookup:
                        print(f"X distance closest > WX_DST ({x.distance_closest} > {wx_dst})")
                        print("Setting W as closest")
                    x.closest = (w.id, wx_dst)
                else:
                    # This is the case requires recomputation of the closest and
                    # therefore the reset of the current closest
                    x.closest = (None, np.inf)
                    if w.id == w_lookup and x.id == x_lookup:
                        print(f"X distance closest < WX_DST ({x.distance_closest} < {wx_dst})")
                        print("Recomputing X closest from the tree")
                    x.nearest_from_tree()
                if w.id == w_lookup and x.id == x_lookup:
                    print(f"X: {x.min_str()}")
            elif x.distance_closest > wx_dst:
                if w.id == w_lookup and x.id == x_lookup:
                    print(f"X distance closest > WX_DST ({x.distance_closest} > {wx_dst})")
                    print("Setting W as closest")
                x.closest = (w.id, wx_dst)
                if w.id == w_lookup and x.id == x_lookup:
                    print(f"X: {x.min_str()}")


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
    def queue(self) -> Union[TfClusters, None]:
        return self.__queue

    @property
    def clusters(self) -> Union[TfClusters, None]:
        return self.__clusters

    @property
    def tree(self) -> Union[KDTree, None]:
        return self.__tree

    @property
    def dimension(self) -> int:
        """dimension.

        Returns the dimension of the data points

        Parameters
        ----------

        Returns
        -------
        int
            The dimension of the data points
        """
        return self.__dimension
