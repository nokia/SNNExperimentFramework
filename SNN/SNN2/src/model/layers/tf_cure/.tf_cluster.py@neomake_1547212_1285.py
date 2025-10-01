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

import ctypes
import numpy as np
from typing_extensions import Self
from typing import List, Tuple, Type, Union
from scipy.spatial import KDTree
from SNN2.src.contextManagers.contextManagers import timeit_cnt
import tensorflow as tf

from .tree import tf_rtree

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[2, ], dtype=tf.int32),
                              tf.TensorSpec(shape=[], dtype=tf.int32)))
def glob_dst(local: tf.Tensor,
             remote: tf.Tensor,
             local_repeater: tf.Tensor,
             remote_repeater: tf.Tensor,) -> float:
    local_rep = tf.tile(local, local_repeater)
    remote_rep = tf.repeat(remote, repeats=remote_repeater, axis=0)
    dst = tf.norm(tf.math.subtract(local_rep, remote_rep), axis=-1)
    return tf.reduce_min(dst)
#
#
# class TfCluster:
#     """TfCluster.
#
#     Manage a single tensorflow cluster instance
#     """
#
#     def __init__(self, id: int,
#                  points: tf.Tensor,
#                  n_rep: int = 5,
#                  dtype_pt: Type = tf.float32,
#                  compression_factor: float = 0.5,
#                  representors_subset: Union[tf.Tensor, None] = None,
#                  tree: Union[index.Index, None] = None) -> None:
#         self.__id = id
#         self.__points = points
#         self.pt_dtype = dtype_pt
#         self.compression_factor = compression_factor
#         self.__points = tf.cast(self.__points, dtype=self.pt_dtype)
#         self.__required_rep = n_rep
#
#         self.__distance_closest: Union[float, None] = None
#         self.__closest: Union[int, None] = None
#         self.__representors: Union[tf.Tensor, None] = None
#         self.__reverse_closests: List[Self] = []
#
#         representors_subset = self.__points if representors_subset is None else representors_subset
#         self.compute_representors(subset=representors_subset)
#         self.tree = tree
#         if self.tree is not None:
#             self.insert_into_tree()
#
#     def compute_representors(self, subset: tf.Tensor) -> None:
#         if subset.shape[0] < self.__required_rep:
#             self.__representors = subset
#         else:
#             with timeit_cnt("Full representors cycle"):
#                 self.__representors = self.__full_representors_cycle(subset)
#         self.__representors += self.compression_factor * (tf.stack([self.mean]*self.rep.shape[0]) - self.rep)
#
#     def __full_representors_cycle(self, subset: tf.Tensor) -> tf.Tensor:
#         i = 1
#         tmp_rep = None
#
#         @tf.function(reduce_retracing=True)
#         def __rep_euclidean_dst(adversaries: tf.Tensor) -> tf.Tensor:
#             dst = tf.norm(tf.math.subtract(subset, adversaries), axis=-1)
#             return dst
#
#         while i <= self.__required_rep:
#             adv = tf.expand_dims(self.mean, 0) if tmp_rep is None else tmp_rep
#             diff = self.__required_rep - adv.get_shape().as_list()[0] + 1
#             rep = tf.range(adv.get_shape().as_list()[0])
#             rep = tf.where(rep == 0, diff, 1)
#             adv = tf.repeat(adv, repeats=rep, axis=0)
#             all_dst = tf.map_fn(__rep_euclidean_dst, adv)
#             min_dst = tf.reduce_min(all_dst, axis=0)
#             max_point = tf.gather(subset, tf.math.argmax(min_dst))
#             exp_max_p = tf.expand_dims(max_point, 0)
#             tmp_rep = exp_max_p if tmp_rep is None else tf.concat([tmp_rep, exp_max_p], 0)
#
#             i += 1
#
#         return tmp_rep
#
#     @classmethod
#     def merge(cls, x: Self, y: Self) -> Self:
#         points = tf.concat([x.__points, y.__points], 0)
#         w = cls(x.__id, points,
#                 n_rep = x.__required_rep,
#                 dtype_pt = x.pt_dtype,
#                 compression_factor = x.compression_factor,
#                 tree = x.tree,
#                 representors_subset=tf.concat([x.rep, y.rep], 0))
#         return w
#
#     @property
#     def rep(self) -> tf.Tensor:
#         assert self.__representors is not None, "Representors list not computed yet"
#         return self.__representors
#
#     @property
#     def closest(self) -> int:
#         assert self.__closest is not None, "Cluster closest not yet computed"
#         return self.__closest
#         # return ctypes.cast(self.__closest, ctypes.py_object).value
#
#     @property
#     def reverse_closests(self) -> List[Self]:
#         return self.__reverse_closests
#
#     @closest.setter
#     def closest(self, value: Tuple[int, float]) -> None:
#         self.__closest = value[0]
#         self.distance_closest = value[1]
#         # self.closest.reverse_closests.append(self)
#
#     @property
#     def distance_closest(self) -> float:
#         assert self.__distance_closest is not None
#         return self.__distance_closest
#
#     @distance_closest.setter
#     def distance_closest(self, value: float) -> None:
#         self.__distance_closest = value
#
#     @property
#     def points(self) -> tf.Tensor:
#         return self.__points
#
#     @property
#     def id(self) -> int:
#         return self.__id
#
#     def dst(self, x: Self) -> float:
#         tmp_l = self.rep
#         tmp_x = x.rep
#         local_rep = tf.constant([tmp_x.get_shape()[0],1])
#         remote_rep = tf.constant(tmp_l.get_shape()[0])
#         return glob_dst(tmp_l, tmp_x, local_rep, remote_rep)
#
#     @tf.autograph.experimental.do_not_convert
#     def __rep_closest(self, x: Union[tf.Tensor, np.ndarray],
#                       debug_flag: bool = False) -> None:
#         t = 1
#         diff: List[TfCluster] = []
#         while t <= self.__required_rep + 1:
#             if debug_flag:
#                 print(f"t: {t}, {t}/{self.__required_rep+1}")
#             t = t*2 if t*2 < self.__required_rep + 1 else self.__required_rep+1
#             if debug_flag:
#                 print(f"t: {t}")
#             closest: List[TfCluster] = list(self.tree.nearest(x, t, objects="raw"))
#             if debug_flag:
#                 for c in closest:
#                     print(f"Closest: {c.min_str()}")
#             diff: List[TfCluster] = [x for x in closest if x.id != self.id ]
#             if len(diff) > 0:
#                 break
#
#         if len(diff) == 0:
#             raise Exception(f"In {self.__required_rep+1} all the objects had id {self.id}")
#
#         c: TfCluster = diff[0]
#         c_dst = self.dst(c)
#
#         if debug_flag:
#             print(f"Closest: {c.min_str()} - dst: {c_dst}")
#
#         if self.__closest is None or c_dst < self.distance_closest:
#             if debug_flag:
#                 print(f"self.__closest is None or c_dst < self.distance_closest")
#                 print(f"{self.__closest} - {c_dst} < {self.distance_closest}")
#             self.closest = (c.id, c_dst)
#
#     def nearest_from_tree(self, **kwargs) -> None:
#         assert self.tree is not None
#         for r in self.rep.numpy():
#             self.__rep_closest(r, **kwargs)
#         return self.closest
#
#     def __single_remove(self, x: Union[tf.Tensor, np.ndarray]) -> None:
#         self.tree.delete(self.id, x)
#
#     def __single_insert(self, x: Union[tf.Tensor, np.ndarray]) -> None:
#         self.tree.insert(self.id, x, obj=self)
#
#     def remove_from_tree(self) -> None:
#         for r in self.rep.numpy():
#             self.__single_remove(r)
#
#     def insert_into_tree(self) -> None:
#         for r in self.rep.numpy():
#             self.__single_insert(r)
#
#     @property
#     def mean(self) -> tf.Tensor:
#         return tf.cast(tf.reduce_mean(self.__points, axis=0), dtype=self.pt_dtype)
#
#     def __str__(self) -> str:
#         if self.__closest is not None:
#             return f"Cluster id: {self.__id}\nPoints:\n{self.__points}\ncentroid: {self.mean}\nRepresentors: {self.rep}\nClosest cluster id: {self.closest} - dst: {self.distance_closest}"
#         return f"Cluster id: {self.__id}\nPoints:\n{self.__points}\ncentroid: {self.mean}\nRepresentors: {self.rep}\nClosest cluster not computed yet"
#
#     def min_str(self) -> str:
#         if self.__closest is not None:
#             return f"Cluster id: {self.__id} - Closest cluster id: {self.closest} - dst: {self.distance_closest}"
#         return f"Cluster id: {self.__id} - Closest cluster not computed yet"
#
#     def __eq__(self, __o: Self) -> bool:
#         return self.id == __o.id
#
#     def __hash__(self) -> int:
#         return hash(self.id)
#
# class TfClusters():
#     """TfClusters.
#
#     Manage a queue of clusters.
#     To apply operations to the tree, all functions requires the tree object
#     passed as parameter, and it's later returned as part of the output.
#     This is done such that the tree object is not stored in the class.
#     And it's later updated by the caller.
# 	"""
#
#     def __init__(self, points: tf.Tensor,
#                  n_rep: int = 5,
#                  compression_factor: float = 0.5):
#         with timeit_cnt("Cluster list init"):
#             self.__clusters = {i: TfCluster(i, tf.expand_dims(p, 0),
#                                             n_rep=n_rep,
#                                             compression_factor=compression_factor) \
#                                 for i, p in enumerate(points)}
#
#         with timeit_cnt("Closest computation"):
#             self.compute_closest()
#
#         with timeit_cnt("sorting"):
#             self.__sorted_clusters = list(self.__clusters.values())
#             self.apply_sort()
#
#         self.__size: Union[int, None] = len(self.__clusters)
#
#     def apply_sort(self) -> None:
#         self.__sorted_clusters.sort(key=lambda x: x.distance_closest)
#
#     def compute_closest(self) -> None:
#         for c in self.__clusters.values():
#             id = c.nearest_from_tree()
#             self.__clusters[id].reverse_closests.append(c.id)
#
#     def update_reverse_closest(self, c_id, new_closest_id) -> None:
#             self.__clusters[c_id].reverse_closests.append(new_closest_id)
#
#     def pop(self):
#         c = self.__sorted_clusters.pop(0)
#         del self.__clusters[c.id]
#         return c
#
#     def remove(self, obj: TfCluster) -> None:
#         self.__sorted_clusters.remove(obj)
#         del self.__clusters[obj.id]
#
#     def insert(self, obj: TfCluster) -> None:
#         self.__clusters[obj.id] = obj
#         self.__sorted_clusters.append(obj)
#
#     def get(self, idx: Union[int, None] = None) -> TfCluster:
#         idx = 0 if idx is None else idx
#         return self.__clusters[idx]
#
#     @property
#     def iter(self) -> List[TfCluster]:
#         return list(self.__clusters.values())
#
#     @property
#     def size(self) -> int:
#         assert self.__size is not None, "The size has not been computed yet"
#         self.__size = len(self.__clusters)
#         return self.__size
#
#     def __str__(self) -> str:
#         return f"Number of clusters in the queue: {self.__size}"

class tf_clusters():
    """TfClusters.

    Manage a queue of clusters.
    To apply operations to the tree, all functions requires the tree object
    passed as parameter, and it's later returned as part of the output.
    This is done such that the tree object is not stored in the class.
    And it's later updated by the caller.
	"""

    def __init__(self, points: tf.Tensor,
                 n_rep: int = 5,
                 compression_factor: float = 0.5):
        self.n_rep = n_rep
        self.compression_factor = compression_factor

        with timeit_cnt("Cluster init with ragged tensors"):
            # At init each point is a different cluster, all clusters are in
            # a ragged tensor with shape ([n_points, 1, n_features])
            points = tf.expand_dims(points, 1)
            self.__r_clusters = tf.RaggedTensor.from_tensor(points,
                                                            ragged_rank=1)
            self.__r_reps = tf.RaggedTensor.from_tensor(points,
                                                        ragged_rank=1)
            self.__ids = tf.range(points.shape[0])

            # Define the closest tensor as full of -1 at the beginning
            self.__closest = tf.Variable(tf.zeros([points.shape[0]]))

            # Define the distances tensor as all 0s at the beginning
            self.__distances = tf.Variable(tf.zeros([points.shape[0]]))

        self.apply_rep_compression()
        #
        # with timeit_cnt("sorting"):
        #     self.__sorted_clusters = list(self.__clusters.values())
        #     self.apply_sort()
        #
        # self.__size: Union[int, None] = len(self.__clusters)

    def add_rep(self, id: int, rep: tf.Tensor) -> None:
        """add_rep.

        Add the rep to the cluster with the given id.

        Parameters
        ----------
        id : int
            id of the cluster to add the rep, this id will be used to identify
            the cluster
        rep : tf.Tensor
            representor to add to the cluster

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the id is not in the range of the clusters
            If there is no more space for reps in the cluster
        """
        if id >= self.clusters.shape[0]:
            raise ValueError("The id is not in the range of the clusters")
        if self.reps.row_lengths()[id] == self.n_rep:
            raise ValueError("There is no more space for reps in the cluster")
        tmp_tf = tf.concat([self.__r_reps[id], tf.expand_dims(rep, 0)], axis=0)
        self.__r_reps = tf.concat([self.__r_reps[:id],
                                   tf.expand_dims(tmp_tf, 0),
                                   self.__r_reps[id+1:]], axis=0)

    def best_closest(self,
                     id: int,
                     rep: tf.Tensor,
                     tree: tf_rtree) -> Tuple[int, float]:
        """best_closest.

        Compute the closest cluster to the given set of representors.

        Parameters
        ----------
        id : int
            id of the cluster
        rep : tf.Tensor
            representors to compute the closest cluster
        tree : tf_rtree
            tree with all the clusters

        Returns
        -------
        Tuple[int, float]
            id of the closest cluster and the distance from the cluster
        """
        with timeit_cnt("Best closest"):
            closest_ids = tree.nearest(rep, n=2)
        # The closest cluster is usually the second element becuse the first is
        # the cluster itself. Remove from the tensor where == id
        closest_ids = tf.reshape(closest_ids, [-1])
        with timeit_cnt("cleaning ids"):
            tf_mask = closest_ids != id
            # closest_ids = tf.boolean_mask(closest_ids, tf_mask) # Super slow
            closest_ids = tf.gather(closest_ids, tf.where(tf_mask))
            closest_ids = tf.reshape(closest_ids, [-1]) # Ensure flatten 1D shape
            # Remove duplicate ids (possible when multiple representors are used)
            closest_ids = tf.unique(closest_ids).y

            # Compute distance from the remaining ids
            closest_rep = tf.gather(self.reps, closest_ids)

        def dst(x: tf.Tensor) -> float:
            local_rep = 
            remote_rep = tf.constant(rep.get_shape()[0])
            return glob_dst(rep, x, local_rep, remote_rep)

        with timeit_cnt("compute distances"):
            closest_dst = tf.map_fn(dst, closest_rep, dtype=tf.float32)

        # Return the id of the closest cluster with the minimum distance
        min_idx = tf.argmin(closest_dst)
        return closest_ids[min_idx], closest_dst[min_idx]

    def compute_closest(self, tree: tf_rtree) -> None:
        """compute_closest.

        Compute the closest cluster for each cluster in the queue.

        Parameters
        ----------
        tree : tf_rtree
            tree with all the clusters

        Returns
        -------
        None
        """
        # for c in self:
        #     c_id = c[0]
        #     c_rep = c[2]
        #     closest_id, dst = self.best_closest(c_id, c_rep, tree)
        #     self.__closest[c_id].assign(closest_id.numpy())
        #     self.__distances[c_id].assign(dst.numpy())

        # Apply the previous loop as a vectorized function
        self.__closest, self.__distances = tf.map_fn(
            lambda x: self.best_closest(x[0], x[1], tree),
            elems=(self.ids, self.reps),
            dtype=(tf.int32, tf.float32),
            fn_output_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),
                                 tf.TensorSpec(shape=[], dtype=tf.float32)))
        print(self.__closest)
        print(self.__distances)

    def apply_rep_compression(self) -> None:
        """apply_rep_compression.

        Apply the compression factor to the representors.
        """
        self.__r_reps += self.compression_factor * (self.mean - self.reps)

    @property
    def mean(self) -> tf.Tensor:
        """mean.

        Return the mean of all the points.

        Returns
        -------
        tf.Tensor
            mean of all the points
        """
        return tf.reduce_mean(self.__r_clusters, axis=0)

    @property
    def clusters(self) -> tf.RaggedTensor:
        """clusters.

        Return the ragged tensor with all the clusters.

        Returns
        -------
        tf.RaggedTensor
            ragged tensor with all the clusters
        """
        return self.__r_clusters

    @property
    def reps(self) -> tf.RaggedTensor:
        """reps.

        Return the ragged tensor with all the representors.

        Returns
        -------
        tf.RaggedTensor
            ragged tensor with all the representors for each cluster
        """
        return self.__r_reps

    @property
    def ids(self) -> tf.Tensor:
        """ids.

        Return the tensor with all the ids.

        Returns
        -------
        tf.Tensor
            tensor with all the ids for each cluster
        """
        return self.__ids

    def __setitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError("This method is not implemented and a cluster cannot be set")

    def __getitem__(self, idx: int) -> Tuple[float, tf.Tensor, tf.Tensor, float, float]:
        """__getitem__.

        returns all the information about a cluster given
        a specific index.

        Parameters
        ----------
        idx : int
            index of the cluster to return

        Returns
        -------
        Tuple[float, tf.Tensor, tf.Tensor, float, float]
            Returns the cluster id, the cluster itself, the representors,
            the closest cluster id and the distance from the closest cluster
        """
        return self.ids[idx], self.clusters[idx], self.reps[idx], \
                self.__closest[idx], self.__distances[idx]

    def __iter__(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """__iter__.

        Returns an iterator with all the information about the clusters.

        Returns
        -------
        Tuple[float, tf.Tensor, tf.Tensor, float, float]
            Returns the cluster id, the cluster itself, the representors,
            the closest cluster id and the distance from the closest cluster
        """
        for i in range(self.ids.shape[0]):
            yield self[i]
