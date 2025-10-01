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

    def get_idx(self, c_id: Union[int, tf.Tensor]) -> Union[int, tf.Tensor]:
        """get_idx.

        Get the index of the cluster given the id.

        Parameters
        ----------
        c_id : Union[int, tf.Tensor]
            if int the id of the cluster to get the index
            if a tensor then gather the indexes of the clusters

        Returns
        -------
        Union[int, tf.Tensor]
            index of the cluster, if c_id is int then the output is an int
            otherwise a tensor with the indexes.

        Raises
        ------
        ValueError
            If the id is not in the range of the clusters
        """
        if c_id.shape == () or len(c_id) == 1:
            tf_idx = tf.where(self.ids == c_id)
            if tf_idx.shape[0] == 0:
                raise ValueError("The id is not in the range of the clusters")
            return tf_idx[0][0]
        else:
            mask = tf.equal(self.ids[:, tf.newaxis], c_id[tf.newaxis, :])
            tf_idx = tf.where(mask)[: , 0]
            return tf_idx

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
        idx = self.get_idx(id)

        if self.reps.row_lengths()[idx] == self.n_rep:
            raise ValueError("There is no more space for reps in the cluster")
        tmp_tf = tf.concat([self.__r_reps[idx], tf.expand_dims(rep, 0)], axis=0)
        self.__r_reps = tf.concat([self.__r_reps[:idx],
                                   tf.expand_dims(tmp_tf, 0),
                                   self.__r_reps[idx+1:]], axis=0)

    def best_closest(self,
                     c_id: int,
                     rep: Union[tf.Tensor, tf.RaggedTensor],
                     tree: tf_rtree,
                     debug: bool = False) -> Tuple[int, float]:
        """best_closest.

        Compute the closest cluster to the given set of representors.

        Parameters
        ----------
        c_id : int
            c_id of the cluster
        rep : Union[tf.Tensor, tf.RaggedTensor]
            representors to compute the closest cluster
        tree : tf_rtree
            tree with all the clusters
        debug : bool
            flag to enable debug print

        Returns
        -------
        Tuple[int, float]
            id of the closest cluster and the distance from the cluster
        """
        # The representors tensor is ragged on the second dimension, to compute
        # n I need to get the current length on that dimension
        if isinstance(rep, tf.RaggedTensor):
            n_dim = rep.shape[-1]
            rep = tf.reshape(rep.to_tensor(), [-1, n_dim])
        n_reprers = rep.shape[0]
        n_request = n_reprers + 1
        # if debug:
        #     print(f"Compute closest for c_id: {c_id}")
        #     print(f"Rep shape: {rep.shape}")
        #     print(f"Rep: {rep}")
        #     print(f"n_reprers: {n_request}")
        closest_ids = tree.nearest(rep, n=n_request)
        # if debug:
        #     print(f"Closest ids: {closest_ids}")

        # The closest cluster is usually the second element becuse the first is
        # the cluster itself. Remove from the tensor where == id
        closest_ids = tf.reshape(closest_ids, [-1])
        tf_mask = closest_ids != c_id
        # closest_ids = tf.boolean_mask(closest_ids, tf_mask) # Super slow
        closest_ids = tf.gather(closest_ids, tf.where(tf_mask))
        closest_ids = tf.reshape(closest_ids, [-1]) # Ensure flatten 1D shape
        # Remove duplicate ids (possible when multiple representors are used)
        closest_ids = tf.unique(closest_ids).y
        closest_idx = self.get_idx(closest_ids)
        # If closest_idx is an int expand it to a tensor of shape [1]
        if closest_idx.shape == ():
            closest_idx = tf.expand_dims(closest_idx, 0)
        # if debug:
        #     print(f"Closest ids after clean: {closest_ids}")
        #     print(f"Closest idx: {closest_idx}")

        # Compute distance from the remaining ids
        closest_rep = tf.gather(self.reps, closest_idx)

        remote_shape = tf.constant(n_reprers)
        def dst(x: tf.Tensor) -> float:
            return glob_dst(rep, x, tf.constant([x.shape[0],1]), remote_shape)

        #with timeit_cnt("compute distances"):
        closest_dst = tf.map_fn(dst, closest_rep, dtype=tf.float32)

        # Return the c_id of the closest cluster with the minimum distance
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
        print("Remember that map_fn dst is slow")
        self.__closest, self.__distances = tf.map_fn(
            lambda x: self.best_closest(x[0], x[1], tree),
            elems=(self.ids, self.reps),
            dtype=(tf.int32, tf.float32),
            fn_output_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),
                                 tf.TensorSpec(shape=[], dtype=tf.float32)))


    def update_closest(self, new_c_id: int,
                       old_c_id: int,
                       tree: tf_rtree) -> None:
        """update_closest.

        Function to update the closest data structure.
        The computation procceds in 2 steps:
            1) Identify all the clusters which have old_c_id as closest cluster
               and compute the new closest cluster for those object.
               the cluster with new_c_id falls into this category
            2) Compute the distance between each cluster and new_c_id and
               update the closest data structure only if the new distance is
               lower than the current one

        Assumption:
            - The closest cluster of new_c_id is currently old_c_id

        Parameters
        ----------
        new_c_id : int
            id of the new cluster
        old_c_id : int
            id of the old cluster
        tree : tf_rtree
            tree with all the clusters

        Returns
        -------
        None
        """
        # Init, identify the index of the new clsuter:
        # print(f"New c_id: {new_c_id} old c_id: {old_c_id}")
        new_idx = self.get_idx(new_c_id)
        # print(f"New idx: {new_idx}")

        # Step 1, identify all the clusters which have old_c_id as closest
        # cluster and compute the new closest cluster for those object
        s1_mask = self.closest == old_c_id
        s1_idx = tf.where(s1_mask)
        s1_c_id = tf.gather(self.ids, s1_idx)
        s1_c_reps = tf.gather(self.reps, s1_idx)

        s1_closest, s1_distances = tf.map_fn(
            lambda x: self.best_closest(x[0], x[1], tree),
            elems=(s1_c_id, s1_c_reps),
            dtype=(tf.int32, tf.float32),
            fn_output_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),
                                 tf.TensorSpec(shape=[], dtype=tf.float32)))

        # Swap closest[s1_idx] with the new s1_closest and also distances
        self.__closest = tf.tensor_scatter_nd_update(self.closest,
                                                     s1_idx,
                                                     s1_closest)
        self.__distances = tf.tensor_scatter_nd_update(self.distances,
                                                       s1_idx,
                                                       s1_distances)

        # Step 2, compute the distance between each cluster and new_c_id
        # if the new distance is lower than the previous one update the closest
        # and distance with the new one
        rep = self.reps[new_idx]
        remote_shape = tf.constant(rep.shape[0])
        def dst(x: tf.Tensor) -> float:
            return glob_dst(rep, x, tf.constant([x.shape[0],1]), remote_shape)

        new_dst = tf.map_fn(dst, self.reps, dtype=tf.float32)

        # Get the mask of clusters which have a lower new_distance in comparison
        # to the current one, and also which are not the new_c_id
        new_mask = tf.logical_and(new_dst < self.distances, self.ids != new_c_id)

        self.__closest = tf.where(new_mask, new_c_id, self.closest)
        self.__distances = tf.where(new_mask, new_dst, self.distances)

    def apply_rep_compression(self) -> None:
        """apply_rep_compression.

        Apply the compression factor to the representors.
        """
        self.__r_reps += self.compression_factor * (self.mean - self.reps)

    def compute_representors(self, c_id:int) -> None:
        """recompute_representors.

        Recompute the representors for the given cluster.

        Parameters
        ----------
        c_id : int
            id of the cluster to recompute the representors

        Returns
        -------
        None
        """
        idx = self.get_idx(c_id)
        c_points = self.clusters[idx]
        c_rep = self.reps[idx]

        if c_points.shape[0] > self.n_rep:
            with timeit_cnt("Representors selection algorithm"):
                c_points = self.select_representors(c_points)
            raise NotImplementedError("This method has not been implemented yet")

        self.__r_reps = tf.concat([self.reps[:idx],
                                  [c_points],
                                  self.reps[idx+1:]], 0)

    def select_representors(self, points: tf.Tensor) -> tf.Tensor:
        """select_representors.

        Select the representors for the given cluster.

        Parameters
        ----------
        points : tf.Tensor
            points of the cluster

        Returns
        -------
        tf.Tensor
            representors for the cluster
        """
        if points.shape[0] < self.n_rep:
            return points

        i = 1
        tmp_rep = None

        @tf.function(reduce_retracing=True)
        def __rep_euclidean_dst(adversaries: tf.Tensor) -> tf.Tensor:
            dst = tf.norm(tf.math.subtract(points, adversaries), axis=-1)
            return dst

        with timeit_cnt("Full representors cycle"):
            while i <= self.n_rep:
                with timeit_cnt(f"Cycle {i}"):
                    #print(points)
                    with timeit_cnt("Compute adversarial"):
                        adv = tmp_rep if not tmp_rep is None else tf.expand_dims(tf.reduce_mean(points, axis=0), 0)
                    #print(adv)
                    with timeit_cnt("Compute diff"):
                        diff = self.n_rep - adv.get_shape().as_list()[0] + 1
                    #print(diff)
                    with timeit_cnt("Compute rep"):
                        rep = tf.range(adv.get_shape().as_list()[0])
                    #print(rep)
                    with timeit_cnt("Compute rep mask"):
                        rep = tf.where(rep == 0, diff, 1)
                    #print(rep)
                    with timeit_cnt("Repeat adv"):
                        adv = tf.repeat(adv, repeats=rep, axis=0)
                    #print(adv)
                    with timeit_cnt("Compute all dst"):
                        all_dst = tf.map_fn(__rep_euclidean_dst, adv)
                    with timeit_cnt("Compute all dst without map_fn"):
                        points_shape = points.shape
                        adv_shape = adv.shape
                        points_r = tf.repeat(points, repeats=adv_shape[0], axis=0)
                        adv_r = tf.tile(adv, multiples=[points_r.shape[0] // adv_shape[0], 1])
                        dst = tf.norm(tf.math.subtract(points_r, adv_r), axis=-1)
                        dst = tf.reshape(dst, (points_shape[0], adv_shape[0]))
                        dst = tf.transpose(dst)
                    #print(all_dst)
                    with timeit_cnt("Gather best point"):
                        min_dst = tf.reduce_min(all_dst, axis=0)
                        #print(min_dst)
                        max_point = tf.gather(points, tf.math.argmax(min_dst))
                        #print(max_point)
                        exp_max_p = tf.expand_dims(max_point, 0)
                        #print(exp_max_p)
                    with timeit_cnt("Concatenate"):
                        tmp_rep = exp_max_p if tmp_rep is None else tf.concat([tmp_rep, exp_max_p], 0)
                    #print(tmp_rep)
                    i += 1
                # print(i)

        return tmp_rep

    def merge(self, u_id: int, v_id: int) -> None:
        """merge.

        Merge two clusters given their indexes.

        Parameters
        ----------
        u_idx : int
            index of the first cluster to merge
        v_idx : int
            index of the second cluster to merge

        Returns
        -------
        None
        """
        # Merge the clusters points from v_idx into u_idx
        u_idx = self.get_idx(u_id)
        v_idx = self.get_idx(v_id)

        if u_idx == v_idx:
            raise ValueError("Cannot merge the same cluster")

        if u_idx > self.clusters.shape[0] or v_idx > self.clusters.shape[0] or \
                u_idx < 0 or v_idx < 0:
            raise ValueError("The indexes are not in the range of the clusters")

        points = tf.concat([self.clusters[u_idx][:],
                            self.clusters[v_idx][:]], 0)

        # Special case if u_idx is the last or first element is already
        # managed by python vector slicing
        self.__r_clusters = tf.concat([self.clusters[:u_idx],
                                       [points],
                                       self.clusters[u_idx+1:]], 0)

    def __delitem__(self, id: int) -> None:
        """__delitem__.

        Delete a cluster given the id.

        Parameters
        ----------
        id : int
            id of the cluster to delete

        Returns
        -------
        None
        """
        idx = self.get_idx(id)
        self.__r_clusters = tf.concat([self.clusters[:idx],
                                      self.clusters[idx+1:]], 0)
        self.__r_reps = tf.concat([self.reps[:idx],
                                  self.reps[idx+1:]], 0)
        self.__ids = tf.concat([self.ids[:idx],
                               self.ids[idx+1:]], 0)
        self.__closest = tf.concat([self.closest[:idx],
                                   self.closest[idx+1:]], 0)
        self.__distances = tf.concat([self.distances[:idx],
                                     self.distances[idx+1:]], 0)

    @property
    def closest(self) -> tf.Tensor:
        """closest.

        Return the tensor with all the closest clusters.

        Returns
        -------
        tf.Tensor
            tensor with all the closest clusters
        """
        return self.__closest

    @property
    def distances(self) -> tf.Tensor:
        """distances.

        Return the tensor with all the distances from the closest clusters.

        Returns
        -------
        tf.Tensor
            tensor with all the distances from the closest clusters
        """
        return self.__distances

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

    def __getitem__(self, id: int) -> Tuple[float, tf.Tensor, tf.Tensor, float, float]:
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
        idx = self.get_idx(id)
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
            yield self[self.ids[i]]

    @property
    def size(self) -> int:
        """size.

        Return the number of clusters with more than 0 points

        Returns
        -------
        int
            size of the queue
        """
        return self.clusters.shape[0]
