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
cure module
===========

Module built to comptue the CURE clustering algorithm on top
of tensorflow vector, taking advantage of GPU acceleration when possible.
"""

from typing import Union

import numpy as np
import tensorflow as tf

from SNN2.src.contextManagers.contextManagers import timeit_cnt
from SNN2.src.decorators.decorators import timeit
from .tf_cluster import tf_clusters
from .tree import tf_rtree

class TfCURE:
    """TfCURE.

    This class is the entry point to execute the CURE algorithm using tensorflow
	"""


    def __init__(self, data: tf.Tensor,
                 n_clusters: int = 2,
                 n_rep: int = 5,
                 compression_factor: float = 0.5):
        """__init__.
        Initialization method for the TfCURE object
        The dimension of the objects is obtained from the last dimension
        of the data tensor.

        Parameters
        ----------
        data : tf.Tensor
            The data to cluster
        n_clusters : int
            The number of clusters to obtain
        n_rep : int
            The number of representatives to use
        compression_factor : float
            The compression factor to apply for representatives
        """
        self.__clusters: Union[tf.Tensor, None] = None
        self.__tree: Union[index.Index, None] = None

        self.__data: tf.Tensor = data
        self.__dimension: int = self.data.shape[-1]
        self.__n_clusters: int = n_clusters
        self.__n_rep: int = n_rep
        self.__compression_factor: float = compression_factor

        self.__validate_arguments()

    def __validate_arguments(self) -> None:
        assert not tf.equal(tf.size(self.data), 0), \
                f"The size of data points cannot be 0, shape: {self.data.shape}, size: {tf.size(self.data)}"
        assert self.n_clusters > 0, \
                f"The number of clusters must be higher than 0, {self.n_clusters} has been provided"
        assert self.n_rep > 0, \
                f"The number of representatives must be higher than 0, {self.n_rep} has been provided"
        assert self.compression_factor > 0.0, \
                f"The compression factor must be higher than 0, {self.compression_factor} has been provided"

    def process(self) -> None:
        self.init_rtree()
        self.init_queue()
        self.iterate()
        raise Exception("Not implemented")

    def init_rtree(self, **kwargs) -> None:
        self.tree = tf_rtree(leaf_capacity=200,
                             fill_factor=0.8,
                             dimension=self.dimension)

    def init_queue(self) -> None:
        self.clusters = tf_clusters(self.data, n_rep=self.n_rep,
                                    compression_factor=self.compression_factor)
        with timeit_cnt("Inserting clusters in the tree"):
            self.tree.add_clusters(self.clusters.ids, self.clusters.reps)
        with timeit_cnt("Computing closest cluster"):
            self.clusters.compute_closest(self.tree)


    def iterate(self) -> None:
        """iterate."""

        while self.clusters.size > self.n_clusters:
            with timeit_cnt(f"Queue size: {self.clusters.size}/{self.n_clusters} Iteration time"):
                # Get the cluster closest to it's neighbor
                min_dst_idx = tf.argmin(self.clusters.distances, output_type=tf.int32)
                u_id = self.clusters.ids[min_dst_idx]
                u_dst = self.clusters.distances[min_dst_idx]
                v_id = self.clusters.closest[min_dst_idx]

                print(f"U: {u_id} V: {v_id}, distance: {u_dst}")

                #### Merge the clusters ####
                # Add v points to u
                with timeit_cnt("Merge clusters"):
                    self.clusters.merge(u_id, v_id)

                #### Clean data structures ####
                # Remove v rep from the tree
                v_idx = self.clusters.get_idx(v_id)
                self.tree.remove_cluster(v_id, self.clusters.reps[v_idx])
                # Empty v points
                del self.clusters[v_id]
                # Remove u rep from the tree
                u_idx = self.clusters.get_idx(u_id)
                self.tree.remove_cluster(u_id, self.clusters.reps[u_idx])

                #### Update the remaining clusters ####
                # Recocompute u representors
                self.clusters.compute_representors(u_id)

                # Add the new u rep to the tree
                self.tree.add_cluster(u_id, self.clusters.reps[u_idx])

                # Recompute the closest cluster to u
                # Recompute the distances of the clusters BUT ONLY in respect
                # to the new cluster, if the current dst is < than the new one
                # then do not update.
                self.clusters.update_closest(u_id, v_id, self.tree)

    @property
    def data(self) -> tf.Tensor:
        """data.

        Returns the data tensor

        Parameters
        ----------

        Returns
        -------
        tf.Tensor
            The data tensor
        """
        return self.__data

    @property
    def dimension(self) -> int:
        """dimension.

        Returns the number of dimensions for the points in the data tensor

        Parameters
        ----------

        Returns
        -------
        int
            The number of dimensions
        """
        return self.__dimension

    @property
    def n_clusters(self) -> int:
        """n_clusters.

        Returns the number of clusters to obtain

        Parameters
        ----------

        Returns
        -------
        int
            The number of clusters to obtain
        """
        return self.__n_clusters

    @property
    def n_rep(self) -> int:
        """n_rep.

        Returns the number of representatives to use

        Parameters
        ----------

        Returns
        -------
        int
            The number of representatives to use
        """
        return self.__n_rep

    @property
    def compression_factor(self) -> float:
        """compression_factor.

        Returns the compression factor to apply for representatives

        Parameters
        ----------

        Returns
        -------
        float
            The compression factor to apply for representatives
        """
        return self.__compression_factor

    @property
    def tree(self) -> Union[tf_rtree, None]:
        """tree.

        Returns the rtree object

        Parameters
        ----------

        Returns
        -------
        tf_rtree
            The rtree object
        """
        return self.__tree

    @tree.setter
    def tree(self, tree: tf_rtree) -> None:
        """tree.

        Set the rtree object

        Parameters
        ----------
        tree : tf_rtree
            The rtree object

        Returns
        -------
        """
        self.__tree = tree

    @property
    def clusters(self) -> Union[tf.Tensor, None]:
        """clusters.

        Returns current avilable clusters

        Parameters
        ----------

        Returns
        -------
        tf.Tensor
            The clusters
        """
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters: tf_clusters) -> None:
        """clusters.

        Set the current available clusters

        Parameters
        ----------
        clusters : TfClusters
            The clusters

        Returns
        -------
        """
        self.__clusters = clusters



    # def rm_cluster_tree(self, cluster: TfCluster) -> None:
    #     """rm_cluster_tree.
    #
    #     Remove a cluster from the local tree
    #
    #     Parameters
    #     ----------
    #     cluster : TfCluster
    #         The cluster to remove
    #
    #     Returns
    #     -------
    #     """
    #     for rep in cluster.rep.numpy():
    #         self.__tree.delete(cluster.id, rep)
    #
    # def iterate(self) -> None:
    #     assert self.__queue is not None, "Must initialize the Queue before iteration"
    #     assert self.__tree is not None, "Must iterate the Tree before iteration"
    #
    #     while self.__queue.size > self.__number_of_clusters:
    #         # The pop operation also removes u from the queue
    #         with timeit_cnt(f"Queue size: {self.__queue.size}/{self.__number_of_clusters} Iteration time"):
    #             u: TfCluster = self.__queue.pop()
    #             print(f"U: {u.min_str()}")
    #             v: TfCluster = self.__queue.get(u.closest)
    #             print(f"V: {v.min_str()}")
    #             self.__queue.remove(v)
    #
    #             w: TfCluster = TfCluster.merge(u, v)
    #             print(f"W: {w.min_str()}")
    #             print(f"tree analysis")
    #             print(f"Self tree id: {id(self.__tree)}")
    #             print(f"U tree id: {id(u.tree)}")
    #             print(f"V tree id: {id(v.tree)}")
    #             print(f"W tree id: {id(w.tree)}")
    #             u.remove_from_tree()
    #             self.rm_cluster_tree(u)
    #             v.remove_from_tree()
    #             self.rm_cluster_tree(v)
    #             if w.id == 117:
    #                 t = 6
    #                 c = list(self.__tree.nearest(w.rep.numpy()[0], t, objects="raw"))
    #                 for tmp in c:
    #                     print(f"Tmp: {tmp}")
    #                 raise Exception("Stop")
    #
    #             w.insert_into_tree()
    #
    #             if w.id == 117:
    #                 t = 6
    #                 c = list(self.__tree.nearest(w.rep.numpy()[0], t, objects="raw"))
    #                 for tmp in c:
    #                     print(f"Tmp: {tmp}")
    #                 raise Exception("Stop")
    #                 new_id = w.nearest_from_tree(debug_flag=True)
    #             else:
    #                 new_id = w.nearest_from_tree()
    #             print(f"New closest cluster {new_id}")
    #             self.__queue.update_reverse_closest(new_id, w.id)
    #             print(f"{new_id} cluster state: {self.__queue.get(new_id).min_str()}")
    #             self.__recompute_distances(w, u, v)
    #             print(f"{new_id} cluster state: {self.__queue.get(new_id).min_str()}")
    #             print(f"W: {w.min_str()}")
    #
    #             self.__queue.insert(w)
    #             self.__queue.apply_sort()
    #
    # def __recompute_distances(self, w: TfCluster,
    #                           u: TfCluster,
    #                           v: TfCluster) -> None:
    #     assert self.__queue is not None, "Must initialize the Queue before iteration"
    #
    #     w_lookup=4
    #     x_lookup=117
    #     if w.id == w_lookup:
    #         print(f"Recompute distances of {w_lookup}")
    #         print(f"U: {u.min_str()}")
    #         print(f"V: {v.min_str()}")
    #         print(f"W: {w.min_str()}")
    #
    #     u.reverse_closests.remove(v.id)
    #     v.reverse_closests.remove(u.id)
    #     c_list = u.reverse_closests
    #     c_list.extend(v.reverse_closests)
    #
    #     if w.id == w_lookup:
    #         print(f"Closests: {c_list}")
    #
    #     for x in self.__queue.iter:
    #         if w.id == w_lookup and x.id == x_lookup:
    #             print("Comparing W-X")
    #             print(f"X: {x.min_str()}")
    #
    #         wx_dst = w.dst(x)
    #
    #         if w.id == w_lookup and x.id == x_lookup:
    #             print(f"Distance: {wx_dst}")
    #
    #         if x.closest == u.id or x.closest == v.id:
    #             # This closest is not valid anymore but the closest needs to be
    #             # recomputed only if the new distance w-x is worst than before
    #             if x.distance_closest >= wx_dst:
    #                 if w.id == w_lookup and x.id == x_lookup:
    #                     print(f"X distance closest > WX_DST ({x.distance_closest} > {wx_dst})")
    #                     print("Setting W as closest")
    #                 x.closest = (w.id, wx_dst)
    #             else:
    #                 # This is the case requires recomputation of the closest and
    #                 # therefore the reset of the current closest
    #                 x.closest = (None, np.inf)
    #                 if w.id == w_lookup and x.id == x_lookup:
    #                     print(f"X distance closest < WX_DST ({x.distance_closest} < {wx_dst})")
    #                     print("Recomputing X closest from the tree")
    #                 x.nearest_from_tree()
    #             if w.id == w_lookup and x.id == x_lookup:
    #                 print(f"X: {x.min_str()}")
    #         elif x.distance_closest > wx_dst:
    #             if w.id == w_lookup and x.id == x_lookup:
    #                 print(f"X distance closest > WX_DST ({x.distance_closest} > {wx_dst})")
    #                 print("Setting W as closest")
    #             x.closest = (w.id, wx_dst)
    #             if w.id == w_lookup and x.id == x_lookup:
    #                 print(f"X: {x.min_str()}")
    #
    #
    # def cc(self, x: TfCluster,
    #        wdst: float) -> TfCluster:
    #     assert self.__tree is not None, "Must initialize the Tree before iteration"
    #     return self.__tree.closest_conditioned(x, limit=wdst)
