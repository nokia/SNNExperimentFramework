# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
from rtree import index
import tensorflow as tf

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


class TfCluster:
    """TfCluster.

    Manage a single tensorflow cluster instance
    """

    def __init__(self, id: int,
                 points: tf.Tensor,
                 n_rep: int = 5,
                 dtype_pt: Type = tf.float32,
                 compression_factor: float = 0.5,
                 representors_subset: Union[tf.Tensor, None] = None,
                 tree: Union[index.Index, None] = None) -> None:
        self.__id = id
        self.__points = points
        self.pt_dtype = dtype_pt
        self.compression_factor = compression_factor
        self.__points = tf.cast(self.__points, dtype=self.pt_dtype)
        self.__required_rep = n_rep

        self.__distance_closest: Union[float, None] = None
        self.__closest: Union[int, None] = None
        self.__representors: Union[tf.Tensor, None] = None
        self.__reverse_closests: List[Self] = []

        representors_subset = self.__points if representors_subset is None else representors_subset
        self.compute_representors(subset=representors_subset)
        self.tree = tree
        if self.tree is not None:
            self.insert_into_tree()

    def compute_representors(self, subset: tf.Tensor) -> None:
        if subset.shape[0] < self.__required_rep:
            self.__representors = subset
        else:
            with timeit_cnt("Full representors cycle"):
                self.__representors = self.__full_representors_cycle(subset)
        self.__representors += self.compression_factor * (tf.stack([self.mean]*self.rep.shape[0]) - self.rep)

    def __full_representors_cycle(self, subset: tf.Tensor) -> tf.Tensor:
        i = 1
        tmp_rep = None

        @tf.function(reduce_retracing=True)
        def __rep_euclidean_dst(adversaries: tf.Tensor) -> tf.Tensor:
            dst = tf.norm(tf.math.subtract(subset, adversaries), axis=-1)
            return dst

        while i <= self.__required_rep:
            adv = tf.expand_dims(self.mean, 0) if tmp_rep is None else tmp_rep
            diff = self.__required_rep - adv.get_shape().as_list()[0] + 1
            rep = tf.range(adv.get_shape().as_list()[0])
            rep = tf.where(rep == 0, diff, 1)
            adv = tf.repeat(adv, repeats=rep, axis=0)
            all_dst = tf.map_fn(__rep_euclidean_dst, adv)
            min_dst = tf.reduce_min(all_dst, axis=0)
            max_point = tf.gather(subset, tf.math.argmax(min_dst))
            exp_max_p = tf.expand_dims(max_point, 0)
            tmp_rep = exp_max_p if tmp_rep is None else tf.concat([tmp_rep, exp_max_p], 0)

            i += 1

        return tmp_rep

    @classmethod
    def merge(cls, x: Self, y: Self) -> Self:
        points = tf.concat([x.__points, y.__points], 0)
        w = cls(x.__id, points,
                n_rep = x.__required_rep,
                dtype_pt = x.pt_dtype,
                compression_factor = x.compression_factor,
                tree = x.tree,
                representors_subset=tf.concat([x.rep, y.rep], 0))
        return w

    @property
    def rep(self) -> tf.Tensor:
        assert self.__representors is not None, "Representors list not computed yet"
        return self.__representors

    @property
    def closest(self) -> int:
        assert self.__closest is not None, "Cluster closest not yet computed"
        return self.__closest
        # return ctypes.cast(self.__closest, ctypes.py_object).value

    @property
    def reverse_closests(self) -> List[Self]:
        return self.__reverse_closests

    @closest.setter
    def closest(self, value: Tuple[int, float]) -> None:
        self.__closest = value[0]
        self.distance_closest = value[1]
        # self.closest.reverse_closests.append(self)

    @property
    def distance_closest(self) -> float:
        assert self.__distance_closest is not None
        return self.__distance_closest

    @distance_closest.setter
    def distance_closest(self, value: float) -> None:
        self.__distance_closest = value

    @property
    def points(self) -> tf.Tensor:
        return self.__points

    @property
    def id(self) -> int:
        return self.__id

    def dst(self, x: Self) -> float:
        tmp_l = self.rep
        tmp_x = x.rep
        local_rep = tf.constant([tmp_x.get_shape()[0],1])
        remote_rep = tf.constant(tmp_l.get_shape()[0])
        return glob_dst(tmp_l, tmp_x, local_rep, remote_rep)

    @tf.autograph.experimental.do_not_convert
    def __rep_closest(self, x: Union[tf.Tensor, np.ndarray],
                      debug_flag: bool = False) -> None:
        t = 1
        diff: List[TfCluster] = []
        while t <= self.__required_rep + 1:
            if debug_flag:
                print(f"t: {t}, {t}/{self.__required_rep+1}")
            t = t*2 if t*2 < self.__required_rep + 1 else self.__required_rep+1
            if debug_flag:
                print(f"t: {t}")
            closest: List[TfCluster] = list(self.tree.nearest(x, t, objects="raw"))
            if debug_flag:
                for c in closest:
                    print(f"Closest: {c.min_str()}")
            diff: List[TfCluster] = [x for x in closest if x.id != self.id ]
            if len(diff) > 0:
                break

        if len(diff) == 0:
            raise Exception(f"In {self.__required_rep+1} all the objects had id {self.id}")

        c: TfCluster = diff[0]
        c_dst = self.dst(c)

        if debug_flag:
            print(f"Closest: {c.min_str()} - dst: {c_dst}")

        if self.__closest is None or c_dst < self.distance_closest:
            if debug_flag:
                print(f"self.__closest is None or c_dst < self.distance_closest")
                print(f"{self.__closest} - {c_dst} < {self.distance_closest}")
            self.closest = (c.id, c_dst)

    def nearest_from_tree(self, **kwargs) -> None:
        assert self.tree is not None
        for r in self.rep.numpy():
            self.__rep_closest(r, **kwargs)
        return self.closest

    def __single_remove(self, x: Union[tf.Tensor, np.ndarray]) -> None:
        self.tree.delete(self.id, x)

    def __single_insert(self, x: Union[tf.Tensor, np.ndarray]) -> None:
        self.tree.insert(self.id, x, obj=self)

    def remove_from_tree(self) -> None:
        for r in self.rep.numpy():
            self.__single_remove(r)

    def insert_into_tree(self) -> None:
        for r in self.rep.numpy():
            self.__single_insert(r)

    @property
    def mean(self) -> tf.Tensor:
        return tf.cast(tf.reduce_mean(self.__points, axis=0), dtype=self.pt_dtype)

    def __str__(self) -> str:
        if self.__closest is not None:
            return f"Cluster id: {self.__id}\nPoints:\n{self.__points}\ncentroid: {self.mean}\nRepresentors: {self.rep}\nClosest cluster id: {self.closest} - dst: {self.distance_closest}"
        return f"Cluster id: {self.__id}\nPoints:\n{self.__points}\ncentroid: {self.mean}\nRepresentors: {self.rep}\nClosest cluster not computed yet"

    def min_str(self) -> str:
        if self.__closest is not None:
            return f"Cluster id: {self.__id} - Closest cluster id: {self.closest} - dst: {self.distance_closest}"
        return f"Cluster id: {self.__id} - Closest cluster not computed yet"

    def __eq__(self, __o: Self) -> bool:
        return self.id == __o.id

    def __hash__(self) -> int:
        return hash(self.id)

class TfClusters():
    """TfClusters.

    Manage a queue of clusters
	"""


    def __init__(self, points: tf.Tensor,
                 n_rep: int = 5,
                 compression_factor: float = 0.5,
                 tree: Union[index.Index, None] = None ):
        assert tree is not None, "A tree must be provided"
        with timeit_cnt("Cluster list init"):
            self.__clusters = {i: TfCluster(i, tf.expand_dims(p, 0),
                                         n_rep=n_rep,
                                         compression_factor=compression_factor,
                                         tree=tree) for i, p in enumerate(points)}

        with timeit_cnt("Closest computation"):
            self.compute_closest()

        with timeit_cnt("sorting"):
            self.__sorted_clusters = list(self.__clusters.values())
            self.apply_sort()

        self.__size: Union[int, None] = len(self.__clusters)

    def apply_sort(self) -> None:
        self.__sorted_clusters.sort(key=lambda x: x.distance_closest)

    def compute_closest(self) -> None:
        for c in self.__clusters.values():
            id = c.nearest_from_tree()
            self.__clusters[id].reverse_closests.append(c.id)

    def update_reverse_closest(self, c_id, new_closest_id) -> None:
            self.__clusters[c_id].reverse_closests.append(new_closest_id)

    def pop(self):
        c = self.__sorted_clusters.pop(0)
        del self.__clusters[c.id]
        return c

    def remove(self, obj: TfCluster) -> None:
        self.__sorted_clusters.remove(obj)
        del self.__clusters[obj.id]

    def insert(self, obj: TfCluster) -> None:
        self.__clusters[obj.id] = obj
        self.__sorted_clusters.append(obj)

    def get(self, idx: Union[int, None] = None) -> TfCluster:
        idx = 0 if idx is None else idx
        return self.__clusters[idx]

    @property
    def iter(self) -> List[TfCluster]:
        return list(self.__clusters.values())

    @property
    def size(self) -> int:
        assert self.__size is not None, "The size has not been computed yet"
        self.__size = len(self.__clusters)
        return self.__size

    def __str__(self) -> str:
        return f"Number of clusters in the queue: {self.__size}"
