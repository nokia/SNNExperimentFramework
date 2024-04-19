# class TfClusters:
#     """TfClusters.

#     Class used to describe clusters
#     """

#     def __init__(self, clusters: tf.RaggedTensor,
#                  dtype_int: Type = tf.int32,
#                  dtype_pt: Type = tf.float32) -> None:
#         assert len(clusters.shape) == 3, f"The clusters provided must have 3 dimensions, [number of clusters, number of points per cluster, ]"

#         self.clusters = clusters
#         self.clusters = tf.cast(self.clusters, dtype=tf.float32)
#         self.indexes = tf.range(self.clusters.shape[0])
#         self.rep = None
#         self.distances = None

#         self.i_dtype = dtype_int
#         self.pt_dtype = dtype_pt

#     @tf.function(reduce_retracing=True)
#     def __cluster_euclidean_distance(self,
#                                      points: Tuple[tf.RaggedTensor, tf.Tensor]) -> tf.RaggedTensor:
#         cluster = points[0] if not isinstance(points[0], tf.RaggedTensor) else points[0].to_tensor()
#         counter = points[1] if not isinstance(points[1], tf.RaggedTensor) else points[1].to_tensor()

#         # print(counter)

#         dst = tf.vectorized_map(fn=lambda t: tf.norm(tf.math.subtract(cluster, t), axis=-1),
#                                 elems=counter)
#         # print(dst)
#         # dst = tf.reshape(dst, [counter.shape[0], -1])
#         # print(dst)
#         # raise Exception
#         return tf.expand_dims(tf.RaggedTensor.from_tensor(dst, row_splits_dtype=tf.int32), 0)

#     def cluster_to_points_distances(self, points: tf.Tensor,
#                                     distance_function: Optional[Callable] = None) -> tf.Tensor:
#         distance_function = self.__cluster_euclidean_distance if distance_function is None else distance_function

#         assert self.clusters.shape[0] == points.shape[0], "The number of clusters and the number of points does not correspond"
#         assert self.clusters.dtype == points.dtype, f"The dtype of clusters and points must be the same, found {self.clusters.dtype} != {points.dtype}"

#         dst = tf.map_fn(fn=distance_function, elems=(self.clusters, points),
#                         fn_output_signature=tf.RaggedTensorSpec(shape=[1, points.shape[1], None],
#                                                                 dtype=self.pt_dtype,
#                                                                 ragged_rank=2,
#                                                                 row_splits_dtype=self.i_dtype))
#         dst = tf.squeeze(dst, axis=1)
#         return dst

#     def min_reduction(self, t: tf.RaggedTensor, axis: int = 1) -> Tuple[tf.Tensor, tf.Tensor]:
#         return tf.reduce_min(t, axis=axis)

#     @timeit
#     def get_furthest(self, points: tf.Tensor,
#                      reduction_function: Optional[Callable] = None) -> Tuple[tf.Tensor, tf.Tensor]:
#         reduction_function = self.min_reduction if reduction_function is None else reduction_function

#         print(self.clusters)
#         print(points)
#         dst = self.cluster_to_points_distances(points)
#         print(dst)
#         dst = reduction_function(dst)
#         print(dst)

#         # print(tf.math.argmax(dst, axis=1))
#         # Given the Euclidean distance is not possible to have a distance lower than 0.0
#         dst = dst.to_tensor(default_value=-1.0)
#         print(dst)
#         max_idx = tf.reshape(tf.math.argmax(dst, axis=1), [-1, 1])
#         print(max_idx)

#         objs = tf.gather(self.clusters, max_idx, batch_dims=-1)
#         print(objs)
#         # objs = tf.map_fn(fn=lambda t: tf.gather(t[0], t[1]),
#         #                  elems=(self.clusters, max_idx),
#         #                  fn_output_signature=tf.TensorSpec(shape=[None, None],
#         #                                                    dtype=self.pt_dtype))
#         return objs, max_idx

#     @timeit
#     @tf.function(reduce_retracing=True)
#     def rg_tensor_unique(self, t: tf.RaggedTensor) -> tf.RaggedTensor:
#         unique_tf = tf.raw_ops.UniqueV2(x=t.to_tensor(), axis=tf.constant([0]))[0]
#         res = tf.expand_dims(tf.RaggedTensor.from_tensor(unique_tf, row_splits_dtype=tf.int32), 0)
#         return res

#     def compute_all_representors(self, n_representors: int = 1) -> None:
#         # The first representor is always the furthest point form the mean,
#         # so I can use the already defined function
#         rep, _ = self.get_furthest(self.mean)
#         rep = tf.RaggedTensor.from_nested_row_lengths(
#                 tf.reshape(rep, [-1]), [tf.constant([rep.shape[0]]), tf.constant([rep.shape[1]]*rep.shape[0]), tf.constant([rep.shape[2]]*rep.shape[0])])
#         rep = tf.squeeze(rep, axis=0)

#         with timeit_cnt(f"Remaning representors"):
#             for i in range(1, n_representors):
#                 new_rep, _ = self.get_furthest(rep)
#                 rep = tf.concat([rep, new_rep], axis=1)
#                 with timeit_cnt("Unique"):
#                     rep = tf.map_fn(fn=self.rg_tensor_unique,
#                                     elems=rep, fn_output_signature=tf.RaggedTensorSpec(shape=[1, None, None],
#                                                                                        dtype=self.pt_dtype,
#                                                                                        ragged_rank=2,
#                                                                                        row_splits_dtype=self.i_dtype))
#                 with timeit_cnt("Squeeze"):
#                     rep = tf.squeeze(rep, axis=1)
#             raise Exception

#         raise Exception
#         self.rep = rep

#     def apply_compression(self, n_representors: int = 1,
#                           compression_factor: float = 0.5) -> None:
#         # rep_mean = tf.repeat(self.mean, n_representors, axis=1)
#         # self.rep += compression_factor * (rep_mean - self.rep)
#         self.rep = tf.map_fn(fn=lambda t: t[0] + compression_factor * (tf.repeat(t[1], t[0].to_tensor().shape[0], axis=0) - t[0]),
#                              elems=(self.rep, self.mean),
#                              fn_output_signature=tf.RaggedTensorSpec(shape=[None, None],
#                                                                      dtype=self.pt_dtype,
#                                                                      ragged_rank=1,
#                                                                      row_splits_dtype=self.i_dtype))
#         return None

#     def compute_representors(self, n_representors: int = 1,
#                              index: Optional[tf.Tensor] = None,
#                              **kwargs) -> None:
#         if self.rep is None or index is None:
#             # Never computed any representor for any cluster, must compute all
#             with timeit_cnt("Computation"):
#                 self.compute_all_representors(n_representors=n_representors)
#             with timeit_cnt("Compression"):
#                 self.apply_compression(n_representors=n_representors, **kwargs)
#         return None

#     @classmethod
#     def cluster_to_clusters_distance(cls, objs: Tuple[tf.RaggedTensor, tf.RaggedTensor]) -> tf.Tensor:
#         cluster = cls(tf.expand_dims(objs[0], 0))
#         comparators = tf.expand_dims(objs[1], 1)

#         distances = tf.map_fn(fn=cluster.cluster_to_points_distances,
#                               elems=comparators)
#         distances = tf.squeeze(distances, axis=1)
#         distances = tf.reduce_min(distances, axis=[1, 2])
#         return distances

#     def compute_distances(self):
#         assert not self.rep is None
#         print(self.rep)
#         reps_cmp = tf.stack([tf.concat([self.rep[i+1:,:,:], self.rep[:i, :, :]], axis=0) \
#                              for i in range(self.rep.shape[0])], axis=0)
#         dst_args = tf.stack([tf.concat([tf.range(i+1, self.rep.shape[0]), tf.range(0, i)], axis=0) \
#                              for i in range(self.rep.shape[0])], axis=0)
#         dst = tf.map_fn(fn=self.cluster_to_clusters_distance, elems=(self.rep, reps_cmp),
#                         fn_output_signature=tf.TensorSpec(shape=[reps_cmp.shape[1]],
#                                                           dtype=tf.float32))
#         self.distances = dst
#         self.dst_args = dst_args

#         dst_ord_args = tf.argsort(self.distances)
#         self.distances = tf.gather(self.distances, dst_ord_args, batch_dims=-1)
#         self.dst_args = tf.gather(self.dst_args, dst_ord_args, batch_dims=-1)
#         self.closest = self.dst_args[:, 0]

#     @property
#     def mean(self) -> tf.Tensor:
#         return tf.cast(tf.expand_dims(tf.reduce_mean(self.clusters, axis=1), 1), dtype=tf.float32)


