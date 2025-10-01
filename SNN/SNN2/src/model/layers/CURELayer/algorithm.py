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
TfCURE module
=============

This module is used to compute the CURE algorithm on a dataset using tensorflow
"""

from typing import Callable, List, Optional, Tuple, Type, Union
from typing_extensions import Self
import tensorflow as tf
from tensorflow.python.ops.ragged.segment_id_ops import row_splits_to_segment_ids
from SNN2.src.contextManagers.contextManagers import timeit_cnt

from SNN2.src.decorators.decorators import timeit

Clusters = tf.Tensor

class TfCluster:
    """TfCluster.

    Class that manages a single Tensorflow cluster
    """

    def __init__(self,
                 cluster: tf.Tensor,
                 c_id: int,
                 dtype_int: Type = tf.int32,
                 dtype_pt: Type = tf.float32) -> None:
        self.c_id = c_id
        self.i_dtype = dtype_int
        self.pt_dtype = dtype_pt

        self.points = cluster
        self.points = tf.cast(self.points, dtype=self.pt_dtype)
        assert len(self.points.shape) == 2, f"A cluster cannot contain more than 2 dimensions, one for the list of points, and the other for the coordinates of points, found {len(self.points.shape)}, id: {self.c_id}"

        self.rep = None
        self.distances = None
        self.distances_ids = None

    # @tf.function(reduce_retracing=True)
    def __euclidean_dst(self,
                        adversaries: tf.Tensor) -> tf.Tensor:
        return tf.norm(tf.math.subtract(self.points, adversaries), axis=-1)

    def my_dst(self,
               adversaries: Union[tf.Tensor, Self],
               distance_function: Optional[Callable] = None) -> tf.Tensor:
        if isinstance(adversaries, TfCluster):
            return self.__compute_cluster_distance(self.rep, adversaries, distance_function=distance_function)

        assert len(adversaries.shape) == 2, "The dimensions of the adversial points must be 2, one for the list of points, the second with the coordinates of such point"
        assert self.points.shape[1] == adversaries.shape[1], f"Expected th e same number of coordinates, found {self.points.shape[1]} != {adversaries.shape[1]}"

        distance_function = self.__euclidean_dst if distance_function is None else distance_function
        assert distance_function is not None

        return tf.vectorized_map(fn=distance_function, elems=adversaries)

    @classmethod
    def __compute_cluster_distance(cls,
                                   current_reps: Union[tf.Tensor, None],
                                   adversaries: Self,
                                   distance_function: Optional[Callable] = None) -> tf.Tensor:
        assert current_reps is not None
        assert adversaries.rep is not None
        return tf.reduce_min(cls(current_reps, -1).my_dst(adversaries.rep, distance_function=distance_function))

    @tf.function(reduce_retracing=True)
    def min_reduction(self, t: tf.RaggedTensor, axis: int = 0) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.reduce_min(t, axis=axis)

    def get_furthest(self,
                     adversaries: tf.Tensor,
                     reduction_function: Optional[Callable] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        reduction_function = self.min_reduction if reduction_function is None else reduction_function
        assert reduction_function is not None

        dst = self.my_dst(adversaries)
        dst = reduction_function(dst)

        max_idx = tf.math.argmax(dst)
        objs = tf.gather(self.points, max_idx)

        return objs, max_idx

    def compute_representors(self, n_representors: int = 1,
                             compression_factor: float = 0.0) -> None:
        # The first representor is always the furthest point form the mean,
        # so I can use the already defined function
        m = tf.expand_dims(self.mean, 0)
        fur, _ = self.get_furthest(m)
        rep = tf.expand_dims(fur, 0)

        for i in range(1, n_representors):
            fur, _ = self.get_furthest(rep)
            new_rep = tf.expand_dims(fur, 0)
            rep = tf.concat([rep, new_rep], axis=0)
            rep = tf.raw_ops.UniqueV2(x=rep, axis=tf.constant([0]))[0]

        self.rep = rep
        self.rep += compression_factor * (tf.stack([self.mean]*self.rep.shape[0]) - self.rep)

    @property
    def mean(self) -> tf.Tensor:
        return tf.squeeze(tf.cast(tf.reduce_mean(self.points, axis=0), dtype=self.pt_dtype))

    def __str__(self) -> str:
        return f"{self.c_id} - {self.points}"

class TfClusters:

    def __init__(self, clusters) -> None:
        pass

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
        self.__engine: str = engine
        self.__representors: Union[tf.Tensor, None] = None
        self.__means: Union[tf.Tensor, None] = None

        self.__data_points: tf.Tensor = points_set
        self.__number_of_clusters: int = number_of_clusters
        self.__number_of_representatives: int = number_of_representatives
        self.__compression_factor: float = compression_factor

        self.__validate_arguments()

    def process(self) -> None:
        assert self.__engine == "Default", "The only engine supported up to now is 'Default'"
        self.__default_process()

    def __default_process(self) -> None:
        self.create_queue()

    @timeit
    def create_queue(self, data: Union[tf.Tensor, None] = None) -> None:
        data = self.__data_points if data is None else data
        # self.__queue = [cure_cluster(self.__pointer_data[index_point], index_point) for index_point in range(len(self.__pointer_data))]
        self.__queue = TfClusters(tf.RaggedTensor.from_row_lengths(
                values=data,
                row_lengths=tf.ones([data.shape[0]], dtype=tf.int32)))
        with timeit_cnt("Compute_representors"):
            self.__queue.compute_representors(n_representors=self.__number_of_representatives,
                                              compression_factor=self.__compression_factor)
        raise Exception
        self.__queue.compute_distances()

    def __validate_arguments(self) -> None:
        assert not tf.equal(tf.size(self.__data_points), 0), f"The size of data points cannot be 0, shape: {self.__data_points.shape}, sie: {tf.size(self.__data_points)}"
        assert self.__number_of_clusters > 0, f"The number of clusters must be higher than 0, {self.__number_of_clusters} has been provided"
        assert self.__number_of_representatives > 0, f"The number of representatives must be higher than 0, {self.__number_of_representatives} has been provided"
        assert self.__compression_factor >= 0.0, f"The compression factor must be higher or equal to 0, {self.__compression_factor} has been provided"

    @property
    def clusters(self) -> Union[Clusters, None]:
        return self.__clusters


