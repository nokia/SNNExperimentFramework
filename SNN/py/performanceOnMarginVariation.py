from __future__ import annotations
from itertools import combinations, permutations, product, chain

import os
import numpy as np
import argparse
import pandas as pd
import pickle as pkl
import tensorflow as tf
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from tqdm import tqdm
from umap import distances
from pyclustering.cluster.cure import cure
import tensorflow_probability as tfp
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis as scp_mahalanobis

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.io.directories import DirectoryHandler as DH
from SNN2.src.plot.plotter import plotter as plt
from SNN2.src.io.progressBar import pb
from embeddingTSNE import reduction

from typing import Dict, List, Any, Callable, Optional, Tuple, Union

parser = argparse.ArgumentParser(usage="usage: performanceOnMarginVariation.py [options]",
                                 description="Use the script to generate the statistics on the embeddings",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", nargs='+', dest="embedding_files", default="result/statistics.csv",
                    action="store", help="define the embedding input file")
parser.add_argument("-l", "--label", dest="labels", default="none",
                    action="store", help="define the label")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="iteration0",
                    action="store", help="Define the appendix")
parser.add_argument("-i", "--intermidiate", dest="intermidiate", default="/tmp",
                    action="store", help="Define where to save and store intermidiate output as pkl file")
parser.add_argument("--origin", dest="origins", default=None,
                    action="store", help="define the origins input file")
parser.add_argument("--expL", dest="expected_labels", default=None,
                    action="store", help="Define the expected lables file")
parser.add_argument("-O", "--overwrite", dest="overwrite", default=False,
                    action="store_true", help="Activate the flag to overwrite plot files")

def load(file: str) -> Any:
    with open(file, 'rb') as f:
        return pkl.load(f)

def save(file: str, obj: Any) -> None:
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists")
    with open(file, 'wb') as f:
        pkl.dump(obj, f)

def euclidean(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.square(a - b), -1)

def out_drop(x: tf.Tensor, drop_idx: tf.Tensor, axis: int = 0) -> tf.Tensor:
    x = x.numpy()
    idx = drop_idx.numpy()
    x = np.delete(x, idx, axis=axis)
    return tf.convert_to_tensor(x)

def two_samples_mahalanobis(x: tf.Tensor, y: tf.Tensor, cov: tf.Tensor):
    # x = tf.expand_dims(x[342], axis=0)
    # y = tf.expand_dims(y[342], axis=0)
    # print(f"x: {x[342]}\ny: {y[129]}\nCov: {cov}")
    diff = tf.expand_dims(tf.subtract(x, y), axis=1)
    # print(f"Diff: {diff[342]}")
    diff_t = tf.reshape(diff, [*diff.shape[:-2], diff.shape[-1], diff.shape[-2]])
    # print(f"Diff transpose: {diff_t[342]}")
    mahal = tf.reshape(diff @ cov @ diff_t, [-1])
    # print(f"Mahal^2: {mahal[342]}")
    sqrt_mahal = tf.math.sqrt(mahal)
    # print(f"Mahal: {sqrt_mahal[342]}")

    # print(x[342])
    # print(sqrt_mahal[342])
    # raise Exception
    return sqrt_mahal

def mahalanobis(x: tf.Tensor, data: tf.Tensor, cov=None):
    data_mean = tf.math.reduce_mean(data, axis=0)
    data_std = tf.math.reduce_std(data, axis=0)
    zero_std = tf.where(data_std == 0.0)[:, 0]

    if len(zero_std) > 0:
        data = out_drop(data, zero_std, axis=1)
        data_mean = out_drop(data_mean, zero_std)
        data_std = out_drop(data_std, zero_std)
        x = out_drop(x, zero_std, axis=1)

    data = tf.map_fn(lambda tmp: (tmp-data_mean)/data_std, data)
    x = tf.map_fn(lambda tmp: (tmp-data_mean)/data_std, x)

    diff = tf.subtract(x, data_mean)
    diff_t = tf.transpose(diff)
    cov = tfp.stats.covariance(data)
    inv_cov = tf.linalg.inv(cov, adjoint=False, name=None)
    left_part = tf.linalg.matmul(diff, inv_cov)
    mahal = tf.linalg.matmul(left_part, diff_t)
    return tf.math.sqrt(tf.linalg.tensor_diag_part(mahal))

def get_best_dst(dst, dst_p_values,
                 *args,
                 n: Optional[int] = None,
                 p_value_threshold: float = 0.001,
                 p_value_substitute = np.inf,
                 aggregator: Callable = tf.reduce_mean,
                 **kwargs):
    idx = tf.argsort(dst, *args, **kwargs)
    n = idx.shape[1] if n is None else n
    idx = idx[:, :n]
    dst = tf.gather(dst, idx, batch_dims=-1)
    dst_p_values = tf.gather(dst_p_values, idx, batch_dims=-1)

    p_mask = tf.where(dst_p_values < p_value_threshold, False, True)
    ragged_dst = tf.ragged.boolean_mask(dst, mask=p_mask)
    aggregated_dst = aggregator(ragged_dst, axis=1)
    substitute = tf.constant(False, shape=len(aggregated_dst))
    if aggregator == tf.reduce_min:
        substitute = ragged_dst.row_lengths() == 0
    if aggregator == tf.reduce_mean:
        substitute = tf.math.is_nan(aggregated_dst)

    aggregated_dst = tf.where(substitute, p_value_substitute, aggregated_dst)
    return aggregated_dst, p_mask


def get_avg_cluster_dst(dst, dst_p_values,
                        centroid_matrix, centroid_matrix_p_values,
                        *args,
                        n = 3,
                        p_value_threshold: float = 0.001,
                        p_value_default = np.inf,
                        aggregator: Callable = tf.reduce_mean,
                        absorb_power: Optional[tf.Tensor] = None,
                        dim: Optional[int] = None,
                        **kwargs):
    if absorb_power is not None:
        dst = dst*absorb_power
        assert dim is not None
        dst_p_values = tf.convert_to_tensor(1 - chi2.cdf(dst.numpy(), dim))

    idx = tf.argsort(dst, *args, **kwargs)
    idx = idx[:, :n]
    dst = tf.gather(dst, idx, batch_dims=-1)
    dst_p_values = tf.gather(dst_p_values, idx, batch_dims=-1)

    matrix_indexes = None
    for i in range(1, n-1):
        shift_idx = tf.roll(idx, shift=i, axis=1)
        tmp_idx = tf.stack([idx, shift_idx], axis=-1)
        if matrix_indexes is None:
            matrix_indexes = tmp_idx
        else:
            matrix_indexes = tf.concat([matrix_indexes, tmp_idx], axis=-2)
    matrix_indexes = tf.sort(matrix_indexes)
    matrix_indexes, _ = tf.raw_ops.UniqueV2(x=matrix_indexes, axis=[1])
    dst_centroids = tf.gather_nd(centroid_matrix, matrix_indexes)
    dst_centroids_p_values = tf.gather_nd(centroid_matrix_p_values, matrix_indexes)
    avg_dst_centroids = tf.reduce_mean(dst_centroids, axis=1)

    p_mask = tf.where(dst_p_values < p_value_threshold, False, True)
    ragged_dst = tf.ragged.boolean_mask(dst, mask=p_mask)
    aggregated_dst = aggregator(ragged_dst, axis=1)
    if aggregator == tf.reduce_min:
        aggregated_dst = tf.where(ragged_dst.row_lengths() == 0, np.inf, aggregated_dst)
    if aggregator == tf.reduce_mean:
        aggregated_dst = tf.where(tf.math.is_nan(aggregated_dst), np.inf, aggregated_dst)
    # min_dst_p_values = tf.reduce_min(dst_p_values, axis=1)
    # avg_dst = tf.reduce_mean(dst, axis=1)
    # cluster_avg_dst = avg_dst - avg_dst_centroids
    return aggregated_dst, p_mask

def get_best_cluster_dst(dst, dst_p_values,
                         *args,
                         n = 1,
                         **kwargs):
    assert n < dst.shape[1]

    idx = tf.argsort(dst, *args, **kwargs)
    idx = idx[:, :n]
    dst = tf.gather(dst, idx, batch_dims=-1)
    dst_p_values = tf.gather(dst_p_values, idx, batch_dims=-1)
    dst = tf.reshape(dst, -1)
    dst_p_values = tf.reshape(dst_p_values, -1)
    return dst, dst_p_values

def study_positions(df: pd.DataFrame,
                    label: str,
                    *args,
                    x_min: int = 0,
                    x_max: int = 30,
                    y_min: int = 0,
                    y_max: int = 25,
                    obtained_labels = None,
                    all_dst_ap = None,
                    all_dst_an = None,
                    a_emb = None,
                    p_c_emb = None,
                    n_c_emb = None):
    local_dst_ap_p = args[0]
    local_dst_an_n = args[1]
    df = df.reset_index(drop=True)
    u_df = df[df["E-Correctness"] == label].copy()
    sub_u_df = u_df[(u_df["X"] > x_min) & (u_df["X"] < x_max) &\
                    (u_df["Y"] > y_min) & (u_df["Y"] < y_max) ]
    print(sub_u_df)
    idx = sub_u_df.index.values
    for id in idx:
        print(f"Label obtained: {tf.gather(obtained_labels, id)}")
        print(f"Distance AP: {tf.gather(all_dst_ap, id)}")
        print(f"Distance AN: {tf.gather(all_dst_an, id)}")
        # print(f"P values AP: {tf.gather(p_ap, id)}")
        # print(f"P values AN: {tf.gather(p_an, id)}")
        print(f"Min dst AP: {local_dst_ap_p[id]}")
        print(f"Min dst AN: {local_dst_an_n[id]}")
        a = tf.gather(a_emb.emb, id)
        p_c = p_c_emb.emb
        n_c = n_c_emb.emb
        print(f"Anchor embedding: \n{a}")
        print(f"Positive centroids: \n{p_c}")
        print(f"Negative centroids: \n{n_c}")
        # p_inv_cov = p_inv_cov_matrix
        # n_inv_cov = n_inv_cov_matrix
        # print(f"Positive inverse cov matrix: \n{p_inv_cov}")
        # print(f"Negative inverse cov matrix: \n{n_inv_cov}")

        # # k_p = tf.convert_to_tensor([[1.0, 0, 0, 0],
        # #                             [0, 0.8, 0, 0],
        # #                             [0, 0, 0.8, 0],
        # #                             [0, 0, 0, 0.8]])
        # # k_n = tf.convert_to_tensor([[1.0, 0, 0, 0],
        # #                             [0, 0.7, 0, 0],
        # #                             [0, 0, 0.8, 0],
        # #                             [0, 0, 0, 1.3]])
        # # p_inv_cov = p_inv_cov @ k_p
        # # n_inv_cov = n_inv_cov @ k_n
        # # print(f"Positive inverse cov matrix: \n{p_inv_cov}")
        # # print(f"Negative inverse cov matrix: \n{n_inv_cov}")

        # ap = out_drop(a, p_zero_std)
        # an = out_drop(a, n_zero_std)
        # p_c = out_drop(p_c, p_zero_std, axis=1)
        # n_c = out_drop(n_c, n_zero_std, axis=1)

        # a_p = tf.expand_dims(ap-p_c, axis=1)
        # a_n = tf.expand_dims(an-n_c, axis=1)
        # print(f"A-P: \n{a_p}")
        # print(f"A-N: \n{a_n}")
        # a_p_t = tf.reshape(a_p, [*a_p.shape[:-2], a_p.shape[-1], a_p.shape[-2]])
        # a_n_t = tf.reshape(a_n, [*a_n.shape[:-2], a_n.shape[-1], a_n.shape[-2]])
        # print(f"(A-P)T: \n{a_p_t}")
        # print(f"(A-N)T: \n{a_n_t}")
        # # left_part = a_p @ p_inv_cov
        # # print(f"(A-P)xS^-1 = \n{left_part}")
        # mahal_p = tf.reshape(a_p @ p_inv_cov @ a_p_t, [-1])
        # mahal_n = tf.reshape(a_n @ n_inv_cov @ a_n_t, [-1])
        # print(f"(A-N)xS^-1x(A-N)T = \n{mahal_p}")
        # print(f"(A-P)xS^-1x(A-P)T = \n{mahal_n}")
        # mahal_p = tf.math.sqrt(mahal_p)
        # mahal_n = tf.math.sqrt(mahal_n)
        # print(f"D_mahal(A,P) = \n{mahal_p}")
        # print(f"D_mahal(A,N) = \n{mahal_n}")

        # p_p_val = tf.convert_to_tensor(1 - chi2.cdf(mahal_p.numpy(), 2))
        # n_p_val = tf.convert_to_tensor(1 - chi2.cdf(mahal_n.numpy(), 2))

        # print(f"P values D_mahal(A,P): {p_p_val}")
        # print(f"P values D_mahal(A,N): {n_p_val}")
        # return

        # print(f"P Centroids mahalanobis distances: \n{dst_p_centr}")
        # print(f"N Centroids mahalanobis distances: \n{dst_n_centr}")

        # avg_dst_ap_pcluster, dst_ap_pcluster_p_values = get_avg_cluster_dst(tf.expand_dims(mahal_p, axis=0),
        #                                                                     tf.expand_dims(p_p_val, axis=0),
        #                                                                     dst_p_centr,
        #                                                                     p_p_centr)
        # avg_dst_an_ncluster, dst_an_ncluster_p_values = get_avg_cluster_dst(tf.expand_dims(mahal_n, axis=0),
        #                                                                     tf.expand_dims(n_p_val, axis=0),
        #                                                                     dst_n_centr,
        #                                                                     p_n_centr)
        # print(f"Avg best 3 D_mahal(A,P): {avg_dst_ap_pcluster}")
        # print(f"Avg best 3 D_mahal(A,N): {avg_dst_an_ncluster}")
        # raise Exception

class emb_manager:

    def __init__(self, emb_file: Union[FH, str],
                 id: Optional[str] = None) -> None:
        self.emb_file = emb_file
        self.id = "emb_manager" if id is None else id

        if isinstance(self.emb_file, FH):
            self.emb: tf.Tensor = load(emb_file.path)
        else:
            self.emb: tf.Tensor = load(emb_file)
        if self.emb is None:
            raise Exception(f"A valid embedding.pkl file must be provided")

    def reduce_mean(self) -> tf.Tensor:
        return tf.math.reduce_mean(self.emb, axis=0)

    def reduce_std(self) -> tf.Tensor:
        return tf.math.reduce_std(self.emb, axis=0)

    def drop(self, indexes: tf.Tensor, axis=0) -> tf.Tensor:
        x = self.emb.numpy()
        idx = indexes.numpy()
        x = np.delete(x, idx, axis=axis)
        return tf.convert_to_tensor(x)

    @tf.autograph.experimental.do_not_convert
    def normalize(self,
                  mean: tf.Tensor,
                  std: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.map_fn(lambda x: (x-mean)/std, self.emb)

    def compute_KMeans(self,
                       *args,
                       **kwargs) -> tf.Tensor:
        centroid = KMeans(*args, **kwargs).fit(self.emb.numpy()).cluster_centers_
        centroid = tf.convert_to_tensor(centroid)
        return centroid

    def compute_cure(self,
                     *args,
                     **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        cure_instance = cure(self.emb.numpy(), *args, **kwargs)
        cure_instance.process()
        representors = cure_instance.get_representors()
        representors = list(chain(*representors))
        mean = cure_instance.get_means()
        centroid = tf.convert_to_tensor(mean)
        repr = tf.squeeze(tf.convert_to_tensor(representors))
        return centroid, repr

    def compute_centroid(self,
                         *args,
                         save_flag: bool = True,
                         file_name: Optional[str] = None,
                         only_mean: bool = False,
                         **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        if file_name is not None and os.path.exists(file_name):
            merged = load(file_name)
            if only_mean:
                self.centroid = merged
                return self.centroid
            centroid = tf.expand_dims(merged[0, :], axis=0)
            representors = merged[1:, :]
            self.centroid = centroid
            self.representatives = representors
            return centroid, representors

        centroid, repr = self.compute_cure(*args, **kwargs)
        self.centroid: tf.Tensor = centroid
        if len(repr.shape) < 2:
            repr = tf.expand_dims(repr, 0)
        self.representatives: tf.Tensor = repr

        if save_flag and file_name is not None:
            if not only_mean:
                merged = tf.concat([self.centroid, self.representatives], axis=0)
                save(file_name, merged)
            else:
                save(file_name, self.centroid)

        return centroid, repr

    def distance(self, other: tf.Tensor, dst_function: Callable) -> tf.Tensor:
        return dst_function(self.emb, other)

    def distance_adaptive(self, y: tf.Tensor,
                          dst_function: Callable,
                          file_name: Optional[str] = None) -> tf.Tensor:
        if file_name is not None and \
           os.path.exists(file_name):
            return load(file_name)

        dst = None
        for centroid in y:
            centroid = tf.expand_dims(centroid, axis=0)
            d = dst_function(self.emb, tf.repeat(centroid, self.shape()[0], axis=0))
            d = tf.expand_dims(d, axis=0)
            if dst is None:
                dst = d
            else:
                dst = tf.concat([dst, d], axis=0)

        dst = tf.transpose(dst)
        p_values = tf.ones(dst.shape)

        if file_name is not None:
            save(file_name, (dst, p_values))
        return dst, p_values

    def skt_compute_cov_matrix(self,
                               data: Optional[tf.Tensor] = None) -> tf.Tensor:
        data = self.emb if data is None else data
        if data is None:
            raise Exception

        cov_MCD = MinCovDet().fit(data.numpy()).covariance_
        return tf.convert_to_tensor(cov_MCD, dtype=tf.float32)

    def skt_compute_inv_cov_matrix(self,
                                   file_name: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if file_name is not None and \
           os.path.exists(file_name):
            self.inv_cov_matrix, self.cov_matrix, self.zero_std = load(file_name)
            return self.inv_cov_matrix, self.cov_matrix, self.zero_std

        data_mean = self.reduce_mean()
        data_std = self.reduce_std()
        zero_std = tf.where(data_std == 0.0)[:, 0]

        data = self.emb
        if len(zero_std) > 0:
            data = self.drop(zero_std, axis=1)
            data_mean = out_drop(data_mean, zero_std)
            data_std = out_drop(data_std, zero_std)

        cov_matrix = self.skt_compute_cov_matrix(data=data)
        total = tf.reduce_sum(cov_matrix, axis=1)
        assert len(zero_std) == 0
        zero_idx = tf.where(total == 0.0)[:, 0]
        if len(zero_idx) > 0:
            data = self.drop(zero_idx, axis=1)
            cov_matrix = self.skt_compute_cov_matrix(data=data)

        inv_cov = np.linalg.inv(cov_matrix.numpy())
        inv_cov = tf.convert_to_tensor(inv_cov, dtype=tf.float32)

        if file_name is not None:
            save(file_name, (inv_cov, cov_matrix, zero_idx))

        self.inv_cov_matrix = inv_cov
        self.cov_matrix = cov_matrix
        self.zero_std = zero_idx
        return inv_cov, cov_matrix, zero_idx

    def tfp_compute_inv_cov_matrix(self) -> Tuple[tf.Tensor, tf.Tensor]:
        data_mean = self.reduce_mean()
        data_std = self.reduce_std()
        zero_std = tf.where(data_std == 0.0)[:, 0]

        data = self.emb
        if len(zero_std) > 0:
            data = self.drop(zero_std, axis=1)
            data_mean = out_drop(data_mean, zero_std)
            data_std = out_drop(data_std, zero_std)

        # data = tf.map_fn(lambda tmp: (tmp-data_mean)/data_std, data)
        cov = tfp.stats.covariance(data)
        inv_cov = tf.linalg.inv(cov, adjoint=False, name=None)

        return inv_cov, zero_std

    def distance_cov_adaptive(self,
                              y: tf.Tensor,
                              inv_cov: tf.Tensor,
                              dst_function: Callable,
                              file_name: Optional[str] = None,
                              zero_std: Optional[tf.Tensor] = None,
                              mean: Optional[tf.Tensor] = None,
                              std: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        if file_name is not None and \
           os.path.exists(file_name):
            return load(file_name)

        mean = self.reduce_mean() if mean is None else mean
        std = self.reduce_std() if std is None else std
        zero_std = tf.where(std == 0.0)[:, 0] if zero_std is None else zero_std
        dim = self.shape()[-1] - len(zero_std)

        x = self.emb
        if len(zero_std) > 0:
            x = self.drop(zero_std, axis=1)
            y = out_drop(y, zero_std, axis=1)
            mean = out_drop(mean, zero_std)
            std = out_drop(std, zero_std)

        # x = tf.map_fn(lambda tmp: (tmp-mean)/std, x)
        # y = tf.map_fn(lambda tmp: (tmp-mean)/std, y)

        dst = None
        p_values = None
        for centroid in y:
            centroid = tf.expand_dims(centroid, axis=0)
            d = dst_function(x, tf.repeat(centroid, self.shape()[0], axis=0), inv_cov)
            p_val = tf.convert_to_tensor(1 - chi2.cdf(d.numpy(), dim))
            d = tf.expand_dims(d, axis=0)
            p_val = tf.expand_dims(p_val, axis=0)
            if dst is None:
                dst = d
                p_values = p_val
            else:
                dst = tf.concat([dst, d], axis=0)
                p_values = tf.concat([p_values, p_val], axis=0)

        dst = tf.transpose(dst)
        p_values = tf.transpose(p_values)

        if file_name is not None:
            save(file_name, (dst, p_values))

        return dst, p_values

    def to_df(self, label: str = "emb") -> pd.DataFrame:
        d = {"Data": label}
        for i in range(self.emb.shape[1]):
            d[f"X-{i}"] = self.emb[:, i].numpy()
        return pd.DataFrame(d)

    def shape(self) -> tf.TensorShape:
        return self.emb.shape

    def numpy(self) -> np.ndarray:
        return self.emb.numpy()

    def __str__(self):
        return f"Embedding mananger {self.id}: \n {self.emb}"

class statistics:

    def __init__(self, expectedL: tf.Tensor) -> None:
        self.exp_l = expectedL
        self.labels = {"Positive": 0, "Negative": 1, "Undecided": 2}
        self.cm_labels = {"TP": "TP", "FP": "FP", "TN": "TN", "FN": "FN", "U": "U"}

    def compute_labels(self,
                       ap: tf.Tensor,
                       an: tf.Tensor,
                       ap_p_values: Optional[tf.Tensor] = None,
                       an_p_values: Optional[tf.Tensor] = None,
                       margin: float=0.5,
                       p_value_threshold: float = 0.001) -> tf.Tensor:
        # if ap_p_values is not None and an_p_values is not None:
        #     ap = tf.where(ap_p_values < p_value_threshold, np.inf, ap)
        #     an = tf.where(an_p_values < p_value_threshold, np.inf, an)
        outliers_mask = None
        if ap_p_values is not None and an_p_values is not None:
            ap_outliers = tf.where(ap == np.inf, True, False)
            an_outliers = tf.where(an == np.inf, True, False)
            # ap_outliers = tf.where(ap_p_values < p_value_threshold, True, False)
            # an_outliers = tf.where(an_p_values < p_value_threshold, True, False)
            outliers_mask = tf.math.logical_and(ap_outliers, an_outliers)

        difference: tf.Tensor = ap - an
        assert margin >= 0.0
        l = tf.where(difference < -margin, self.labels["Positive"], self.labels["Undecided"])
        l = tf.where(difference > margin, self.labels["Negative"], l)
        if outliers_mask is not None:
            l = tf.where(outliers_mask, self.labels["Undecided"], l)

        return tf.cast(l, tf.int8)

    def eq_l(self, l: tf.Tensor) -> tf.Tensor:
        return tf.where(self.exp_l == l)[:, 0]

    def neq_l(self, l: tf.Tensor) -> tf.Tensor:
        return tf.where(self.exp_l != l)[:, 0]

    def compute_accuracy(self, ap: tf.Tensor, an: tf.Tensor, **kwargs) -> float:
        l = self.compute_labels(ap, an, **kwargs)
        n_eq = len(self.eq_l(l))
        n_neq = len(self.neq_l(l))
        total = n_eq + n_neq
        return n_eq/total

    def compute_correctness(self, ap: tf.Tensor, an: tf.Tensor, **kwargs) -> tf.Tensor:
        l = self.compute_labels(ap, an, **kwargs)
        correctness = tf.where((self.exp_l == self.labels["Positive"]) & \
                               (self.exp_l == l), self.cm_labels["TP"], self.cm_labels["U"])
        correctness = tf.where((self.exp_l == self.labels["Positive"]) & \
                               (self.exp_l != l) & (l != self.labels["Undecided"]),
                               self.cm_labels["FN"], correctness)
        correctness = tf.where((self.exp_l == self.labels["Negative"]) & \
                               (self.exp_l == l), self.cm_labels["TN"], correctness)
        correctness = tf.where((self.exp_l == self.labels["Negative"]) & \
                               (self.exp_l != l) & (l != self.labels["Undecided"]),
                               self.cm_labels["FP"], correctness)
        return correctness

    def compute_cm(self, ap: tf.Tensor, an: tf.Tensor, normalize: bool = False, **kwargs) -> Dict[str, Union[int, float]]:
        l = self.compute_labels(ap, an, **kwargs)
        eq_idx = self.eq_l(l)
        neq_idx = self.neq_l(l)
        p_correct_idx = tf.where(tf.gather(l, eq_idx) == self.labels["Positive"])[:, 0]
        n_correct_idx = tf.where(tf.gather(l, eq_idx) == self.labels["Negative"])[:, 0]
        p_wrong_idx = tf.where(tf.gather(l, neq_idx) == self.labels["Positive"])[:, 0]
        n_wrong_idx = tf.where(tf.gather(l, neq_idx) == self.labels["Negative"])[:, 0]
        u_idx = tf.where(tf.gather(l, neq_idx) == self.labels["Undecided"])[:, 0]

        res = {"TP": len(p_correct_idx),
               "FP": len(p_wrong_idx),
               "TN": len(n_correct_idx),
               "FN": len(n_wrong_idx),
               "U": len(u_idx)}

        if normalize:
            total = len(eq_idx) + len(neq_idx)
            res = {k: v/total for k, v in res.items()}

        return res

    def get_statistics(self, *args,
                       accuracy_kwargs: Optional[Dict[str, Any]] = None,
                       cm_kwargs: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Dict[str, Union[int, float]]:
        accuracy_kwargs = {} if accuracy_kwargs is None else accuracy_kwargs
        cm_kwargs = {} if cm_kwargs is None else cm_kwargs
        accuracy: float = self.compute_accuracy(*args, **accuracy_kwargs, **kwargs)
        cm: Dict[str, Union[int, float]] = self.compute_cm(*args, **cm_kwargs, **kwargs)

        res = {'Accuracy': accuracy}
        res.update(cm)
        res["f1-score"] = 2*res["TP"]/(2*res["TP"] + res["FP"] + res["FN"] + res["U"])
        return res

    def study(self, start, stop,
              *args,
              num: int = 1000,
              **kwargs) -> Tuple[pd.DataFrame, float, float]:
        stats_df = None
        best_m = 0
        best_points = 0
        for m in np.linspace(start, stop, num=num):
            s = self.get_statistics(*args,
                                    margin=m,
                                    **kwargs)
            if s["U"] < 1.0:
                points = round(abs(s["TP"] + s["TN"] - s["FP"] - s["FN"])/(s["TP"] + s["TN"] + s["FP"] + s["FN"] + s["U"]), 4)
                # points = round(abs(s["TP"] + s["TN"] - s["FP"] - s["FN"])/(s["TP"] + s["TN"] + s["FP"] + s["FN"]), 4)
                # points = 1.0 - round(abs(s["TP"] + s["TN"] - s["FP"] - s["FN"])/(s["TP"] + s["TN"] + s["FP"] + s["FN"]), 4)
            else:
                points = 0.0
            if points > best_points:
                best_points = points
                best_m = round(m, 5)

            stat_l = list(s.keys())
            stat_v = list(s.values())
            tmp_df = pd.DataFrame({'Margin': m, 'Statistic': stat_l, 'Value': stat_v})
            stats_df = tmp_df if stats_df is None else pd.concat([stats_df, tmp_df]).reset_index(drop=True)

        if stats_df is None:
            raise Exception

        return stats_df, best_points, best_m

    def __plot_accuracy(self,
                        df: pd.Dataframe,
                        margin: float,
                        output: DH,
                        *args,
                        stat_col: str = "Statistic",
                        stat_req: List[str] = ["Accuracy", "f1-score"],
                        dst_type: str = "Euclidean",
                        label: str = "",
                        appendix: str = "",
                        **kwargs) -> None:
        accuracy_df = df[df[stat_col].isin(stat_req)].copy()
        p = plt(accuracy_df, format=["pdf", "png"])
        p("line", x="Margin", y="Value", hue=stat_col)
        p.plot.axvline(margin, color='black')
        p.set(ylabel="Accuracy [0-1]",
              title=f"[{dst_type}] Difficult dst Accuracy x marging values")
        p.set_legend(ncol=5, y=-0.32)
        p.save(f"{output.path}/{label}_accuracy_evolution_{dst_type}_{appendix}.pdf")

    def __plot_cm(self,
                  df: pd.Dataframe,
                  margin: float,
                  output: DH,
                  *args,
                  stat_col: str = "Statistic",
                  stat_req: List[str] = ["TP", "FP", "TN", "FN", "U"],
                  dst_type: str = "Euclidean",
                  label: str = "",
                  appendix: str = "",
                  margin_round: int = 4,
                  study_max: float = 10.0,
                  **kwargs) -> None:
        cm_df = df[df[stat_col].isin(stat_req)].copy()
        p = plt(cm_df, format=["pdf", "png"])
        p("lineStack", x="Margin", y="Value", hue=stat_col)

        def insert_margin(m, y):
            p.plot.axvline(m, color='black')
            margin = round(m, margin_round)
            cm_df["SubMargin"] = round(cm_df["Margin"] - m, 4)
            best_cm = cm_df[cm_df["SubMargin"] == cm_df["SubMargin"].abs().min()]
            best_dict = {}
            for _, row in best_cm.iterrows():
                best_dict[row["Statistic"]] = row["Value"]
            txt = f"TP: {best_dict['TP']}, TN: {best_dict['TN']}\nFP: {best_dict['FP']}, FN: {best_dict['FN']}\nU: {best_dict['U']}, Margin: {m}"
            p.plot.text(m+(0.015*study_max), y, txt, fontsize=9)

        insert_margin(margin, .8)
        # insert_margin(0.5, .6)
        # insert_margin(1.0, .4)

        p.set(ylabel="CM portion [0-1]",
              xlabel="Margin",
              title=f"[{dst_type}] Difficult dst Accuracy x marging values")
        p.set_legend(ncol=5, y=-0.32)
        p.save(f"{output.path}/{label}_cm_evolution_{dst_type}_{appendix}.pdf")

    def plot(self, df: pd.Dataframe,
             points: float,
             margin: float,
             output: DH,
             *args,
             accuracy_kwargs: Optional[Dict[str, Any]] = None,
             cm_kwargs: Optional[Dict[str, Any]] = None,
             **kwargs) -> None:
        accuracy_kwargs = {} if accuracy_kwargs is None else accuracy_kwargs
        cm_kwargs = {} if cm_kwargs is None else cm_kwargs

        p_bar = pb.bar(total=(2))
        self.__plot_accuracy(df, margin, output, *args, **accuracy_kwargs, **kwargs)
        p_bar.update(1)

        self.__plot_cm(df, margin, output, *args, **cm_kwargs, **kwargs)
        p_bar.update(1)
        p_bar.close()

class umap_study:

    @classmethod
    def set_kwargs(cls,
                   default: Dict[str, Any],
                   others: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        others = {} if others is None else others
        default.update(others)
        return default

    @classmethod
    def get_filename(cls,
                     output: DH,
                     prefix: str = "file_name",
                     appendix: str = "",
                     extension: str = "pdf") -> str:
        return f"{output.path}/{prefix}{appendix}.{extension}"

    @classmethod
    def conclude(cls, p: plt,
                 output: DH,
                 legend_kw: Optional[Dict[str, Any]] = None,
                 prefix: str = "file_name",
                 p_bar: Optional[tqdm] = None,
                 appendix: str = "") -> None:
        legend_default = {"x": 0.5, "y": -0.32, "ncol": 5}
        legend_kwargs = cls.set_kwargs(legend_default, legend_kw)
        p.set_legend(**legend_kwargs)
        p.save(cls.get_filename(output, prefix=prefix, appendix=appendix))
        if p_bar is not None:
            p_bar.update(1)

    @classmethod
    def scatter(cls,
                umap_df: pd.DataFrame,
                output: DH,
                *args,
                overwrite: bool = False,
                legend_kwargs: Optional[Dict[str, Any]] = None,
                dst_type: str = "Euclidean",
                title: str = "Scatter",
                prefix: str = "umap_representation_Scatter",
                appendix: str = "",
                p_bar: Optional[tqdm] = None,
                **kwargs) -> None:
        if not overwrite and os.path.exists(cls.get_filename(output, prefix=f"{dst_type}_{prefix}", appendix=f"_{appendix}")):
            if p_bar is not None:
                p_bar.update(1)
            return

        p = plt(umap_df, format=["pdf", "png"])
        p("joint", x="X", y="Y", kind="scatter", **kwargs)
        p.plot.fig.suptitle(f"[{dst_type}] {title}")
        cls.conclude(p, output, legend_kw=legend_kwargs,
                     prefix=f"{dst_type}_{prefix}",
                     appendix=f"_{appendix}", p_bar=p_bar)

    @classmethod
    def ecdf(cls,
             umap_df: pd.DataFrame,
             output: DH,
             *args,
             overwrite: bool = False,
             legend_kwargs: Optional[Dict[str, Any]] = None,
             dst_type: str = "Euclidean",
             title: str = "ECDF",
             prefix: str = "ecdf",
             appendix: str = "",
             p_bar: Optional[tqdm] = None,
             **kwargs) -> None:
        if not overwrite and os.path.exists(cls.get_filename(output, prefix=f"{dst_type}_{prefix}", appendix=f"_{appendix}")):
            if p_bar is not None:
                p_bar.update(1)
            return

        p = plt(umap_df, format=["pdf", "png"])
        p("ecdf", **kwargs)
        p.set(title=f"[{dst_type}] {title}")
        cls.conclude(p, output, legend_kw=legend_kwargs,
                     prefix=f"{dst_type}_{prefix}", appendix=f"_{appendix}", p_bar=p_bar)

    @classmethod
    def kde(cls,
            umap_df: pd.DataFrame,
            output: DH,
            *args,
            overwrite: bool = False,
            legend_kwargs: Optional[Dict[str, Any]] = None,
            prefix: str = "kde",
            appendix: str = "data_umap_representation",
            dst_type: str = "Euclidean",
            scatter_df: Optional[pd.DataFrame] = None,
            scatter_kwargs: Optional[Dict[str, Any]] = None,
            p_bar: Optional[tqdm] = None,
            title: str = "UMAP KDE representation",
            **kwargs) -> None:
        if not overwrite and os.path.exists(cls.get_filename(output, prefix=f"{dst_type}_{prefix}", appendix=f"_{appendix}")):
            if p_bar is not None:
                p_bar.update(1)
            return

        kde_default = {"fill": True, "thresh": 0.01, "alpha": .8, "x": "X", "y": "Y",
                       "marginal_kws": {'common_norm': False}}
        kde_default = cls.set_kwargs(kde_default, kwargs)

        p = plt(umap_df, format=["pdf", "png"])
        p("joint", kind="kde", **kde_default)

        if scatter_df is not None:
            sct_ax = sns.scatterplot(data=scatter_df, x="X", y="Y", ax=p.plot.ax_joint, **scatter_kwargs)

        p.plot.fig.suptitle(f"[{dst_type}] {title}")
        cls.conclude(p, output, legend_kw=legend_kwargs,
                     prefix=f"{dst_type}_{prefix}",
                     appendix=f"_{appendix}", p_bar=p_bar)

    @classmethod
    def sub_kde(cls, *args,
                scatter_df: Optional[pd.DataFrame] = None,
                sub_column: str = "",
                sub_labels: List[List[str]] = [[]],
                appendix: str = "",
                **kwargs) -> None:
        assert scatter_df is not None
        assert sub_column in scatter_df.columns

        for labels in sub_labels:
            tmp_scatter_df = scatter_df[scatter_df[sub_column].isin(labels)]
            cls.kde(*args, scatter_df=tmp_scatter_df,
                    title=f"UMAP KDE representation [{'-'.join(labels)}]",
                    appendix=f"{'_'.join(labels)}_{appendix}", **kwargs)

def main():
    options = parser.parse_args()
    inputs: List[FH] = [FH(f, create=False) for f in options.embedding_files]
    label: str = options.labels
    origin: FH = FH(options.origins, create=False)
    expected_labels: FH = FH(options.expected_labels, create=False)
    output: DH = DH(options.output_folder, create=False)
    appendix: str = options.appendix
    intermidiate: str = options.intermidiate
    overwrite: bool = options.overwrite

    assert len(inputs) == 3

    p_emb: emb_manager = emb_manager(inputs[0], id="P")
    a_emb: emb_manager = emb_manager(inputs[1], id="A")
    n_emb: emb_manager = emb_manager(inputs[2], id="N")

    p_centr_file = f"{intermidiate}/p_centroid_{appendix}.pkl"
    n_centr_file = f"{intermidiate}/n_centroid_{appendix}.pkl"

    n_clusters = 1
    n_repr = 1
    p_emb.compute_centroid(n_clusters, file_name=p_centr_file,
                           number_represent_points=n_repr,
                           compression=0.5, only_mean=True)
    n_emb.compute_centroid(n_clusters, file_name=n_centr_file,
                           number_represent_points=n_repr,
                           compression=0.5, only_mean=True)
    p_c_emb: emb_manager = emb_manager(FH(p_centr_file, create=False))
    n_c_emb: emb_manager = emb_manager(FH(n_centr_file, create=False))


    exp_l: tf.Tensor = load(expected_labels.path)
    stats: statistics = statistics(exp_l)

    ap_dst_E_file = f"{intermidiate}/ap_centroid_distances_euclidean_{appendix}.pkl"
    an_dst_E_file = f"{intermidiate}/an_centroid_distances_euclidean_{appendix}.pkl"
    ap_dst_M_file = f"{intermidiate}/ap_centroid_distances_mahalanobis_{appendix}.pkl"
    an_dst_M_file = f"{intermidiate}/an_centroid_distances_mahalanobis_{appendix}.pkl"
    p_inv_cov_matrix_file = f"{intermidiate}/p_inv_cov_matrix_{appendix}.pkl"
    n_inv_cov_matrix_file = f"{intermidiate}/n_inv_cov_matrix_{appendix}.pkl"

    p_emb.skt_compute_inv_cov_matrix(file_name=p_inv_cov_matrix_file)
    n_emb.skt_compute_inv_cov_matrix(file_name=n_inv_cov_matrix_file)

    dst_args = (two_samples_mahalanobis, )
    p_dst_kwargs = {"zero_std" : p_emb.zero_std,
                    "mean": p_emb.reduce_mean(),
                    "std": p_emb.reduce_std()}
    n_dst_kwargs = {"zero_std" : n_emb.zero_std,
                    "mean": n_emb.reduce_mean(),
                    "std": n_emb.reduce_std()}
    all_dst_M_ap, all_p_M_ap = a_emb.distance_cov_adaptive(p_c_emb.emb,
                                                           p_emb.inv_cov_matrix,
                                                           file_name=ap_dst_M_file,
                                                           *dst_args, **p_dst_kwargs)
    all_dst_M_an, all_p_M_an = a_emb.distance_cov_adaptive(n_c_emb.emb,
                                                           n_emb.inv_cov_matrix,
                                                           file_name=an_dst_M_file,
                                                           *dst_args, **n_dst_kwargs)
    all_dst_E_ap, all_p_E_ap = a_emb.distance_adaptive(p_c_emb.emb, euclidean,
                                                       file_name=ap_dst_E_file)
    all_dst_E_an, all_p_E_an = a_emb.distance_adaptive(n_c_emb.emb, euclidean,
                                                       file_name=an_dst_E_file)

    ap_p_val_threshold = 0.001
    an_p_val_threshold = 0.001
    # p_absorbtion_power = tf.convert_to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # n_absorbtion_power = tf.convert_to_tensor([1.0, 0.5, 1.0, 0.5])
    dst_M_ap, p_M_ap = get_best_dst(all_dst_M_ap, all_p_M_ap,
                                    n=3,
                                    aggregator=tf.reduce_min,
                                    p_value_threshold=ap_p_val_threshold)
    dst_M_an, p_M_an = get_best_dst(all_dst_M_an, all_p_M_an,
                                    n=3,
                                    aggregator=tf.reduce_min,
                                    p_value_threshold=an_p_val_threshold)
    dst_E_ap, p_E_ap = get_best_dst(all_dst_E_ap, all_p_E_ap,
                                    n=3,
                                    aggregator=tf.reduce_min,
                                    p_value_threshold=ap_p_val_threshold)
    dst_E_an, p_E_an = get_best_dst(all_dst_E_an, all_p_E_an,
                                    n=3,
                                    aggregator=tf.reduce_min,
                                    p_value_threshold=an_p_val_threshold)
    print(f"d_m(A,P) < p_value {ap_p_val_threshold}: {tf.where(dst_M_ap == np.inf)[:, 0].shape[0]}")
    print(f"d_m(A,N) < p_value {an_p_val_threshold}: {tf.where(dst_M_an == np.inf)[:, 0].shape[0]}")
    print(f"d_e(A,P) < p_value {ap_p_val_threshold}: {tf.where(dst_E_ap == np.inf)[:, 0].shape[0]}")
    print(f"d_e(A,N) < p_value {an_p_val_threshold}: {tf.where(dst_E_an == np.inf)[:, 0].shape[0]}")

    exp_l: tf.Tensor = load(expected_labels.path)

    stats: statistics = statistics(exp_l)
    E_labels = stats.compute_labels(dst_E_ap, dst_E_an)
    M_labels = stats.compute_labels(dst_M_ap, dst_M_an,
                                    ap_p_values=p_M_ap, an_p_values=p_M_an)

    study_min = 0.0
    study_max = 10
    study_max_E = 4.0
    stats_E_df, E_points, E_best_m = stats.study(study_min, study_max_E,
                                                 dst_E_ap, dst_E_an,
                                                 cm_kwargs={'normalize': True})
    stats_M_df, M_points, M_best_m = stats.study(study_min, study_max,
                                                 dst_M_ap, dst_M_an,
                                                 cm_kwargs={'normalize': True},
                                                 ap_p_values=p_M_ap,
                                                 an_p_values=p_M_an)
    M_best_m = 0.5
    E_best_m = 0.5
    print(f"d_m, The best margin {M_best_m} has obtained: {M_points} points")
    print(f"d_e, The best margin {E_best_m} has obtained: {E_points} points")
    margin_round = 4
    stats_M_df = stats_M_df.round(margin_round)
    stats_E_df = stats_E_df.round(margin_round)

    stats.plot(stats_M_df, M_points, M_best_m, output,
               cm_kwargs={"margin_round": margin_round,
                          "study_max": study_max},
               dst_type = "Mahalanobis", label = label, appendix = appendix)
    stats.plot(stats_E_df, E_points, E_best_m, output,
               cm_kwargs={"margin_round": margin_round,
                          "study_max": study_max},
               label = label, appendix = appendix)

    intermidiate_file = f"{intermidiate}/umap_df_{appendix}.pkl"
    centroid_intermidiate_file = f"{intermidiate}/umap_centroids_df_{appendix}.pkl"
    paths = [input.path for input in inputs]
    paths.extend([p_centr_file, n_centr_file])
    labels = ["Positive", "Anchor", "Negative", "p_centroids", "n_centroids"]
    centroid_studies_paths = [input.path for input in inputs]
    centroid_studies_label = ["Positive", "Anchor", "Negative"]
    # for c_files, repr in zip(enumerate(centroid_studies_files), range(2, 20, 2)):
    #     c_files = c_files[1]
    #     p_f = c_files[0]
    #     n_f = c_files[1]
    #     centroid_studies_paths.extend([p_f, n_f])
    #     centroid_studies_label.extend([f"p_centroids-{repr}", f"n_centroids-{repr}"])

    full_dim = pd.DataFrame()
    for path, label in zip(paths, labels):
        emb = emb_manager(path)
        emb_df = emb.to_df(label=label)
        full_dim = pd.concat([full_dim, emb_df])

    umap_df = reduction(paths, labels, intermidiate=intermidiate_file)
    # umap_centroids_study_df = reduction(centroid_studies_paths, centroid_studies_label, intermidiate=centroid_intermidiate_file)

    umap_gb_df = umap_df[umap_df["Data"].isin(["Positive", "Negative"])].copy()
    umap_diff_df = umap_df[umap_df["Data"] == "Anchor"].copy()
    umap_centroid_df = umap_df[umap_df["Data"].isin(["p_centroids", "n_centroids"])]

    # study_centroid_gb_df = umap_centroids_study_df[umap_centroids_study_df["Data"].isin(["Positive", "Negative"])].copy()
    # study_centroid_dfs = [
    #             umap_centroids_study_df[umap_centroids_study_df["Data"].isin([f"p_centroids-{i}", f"n_centroids-{i}"])] \
    #             for i in range(2, 20, 2)
    #         ]

    full_gb_df = full_dim[full_dim["Data"].isin(["Positive", "Negative"])].copy()
    full_diff_df = full_dim[full_dim["Data"] == "Anchor"].copy()
    full_centroid_df = full_dim[full_dim["Data"].isin(["p_centroids", "n_centroids"])]

    M_diff_correctness = stats.compute_correctness(dst_M_ap, dst_M_an, margin=M_best_m,
                                                   ap_p_values=p_M_ap, an_p_values=p_M_an).numpy()
    M_diff_correctness = np.array([x.decode() for x in M_diff_correctness])
    E_diff_correctness = stats.compute_correctness(dst_E_ap, dst_E_an, margin=E_best_m,
                                                   ap_p_values=p_E_ap, an_p_values=p_E_an).numpy()
    E_diff_correctness = np.array([x.decode() for x in E_diff_correctness])

    umap_diff_df["M-Correctness"] = M_diff_correctness
    full_diff_df["M-Correctness"] = M_diff_correctness
    full_diff_df["d_m(P-cluster)"] = dst_M_ap
    full_diff_df["d_m(N-cluster)"] = dst_M_an
    umap_diff_df["E-Correctness"] = E_diff_correctness
    full_diff_df["E-Correctness"] = E_diff_correctness
    full_diff_df["d_e(P-cluster)"] = dst_E_ap
    full_diff_df["d_e(N-cluster)"] = dst_E_an


    study_positions(umap_diff_df, "FN",
                    dst_E_ap, dst_E_an,
                    x_min=-10, x_max=-4, y_min=-6, y_max=0,
                    obtained_labels=E_labels,
                    all_dst_ap = all_dst_E_ap,
                    all_dst_an = all_dst_E_an,
                    a_emb = a_emb,
                    p_c_emb = p_c_emb,
                    n_c_emb = n_c_emb)
    # study_positions(diff_df, "TP", avg_dst_ap_pcluster, avg_dst_an_ncluster, x_min=10, x_max=20, y_min=5, y_max=10)
    # study_positions(diff_df, "FP", avg_dst_ap_pcluster, avg_dst_an_ncluster, x_min=-8, x_max=-2, y_min=0, y_max=6)
    raise Exception

    # for ctr_df, repr in zip(study_centroid_dfs, range(2, 20, 2)):
    #     umap_study.kde(study_centroid_gb_df, output, hue="Data", appendix=f"Centroids-{repr}_diffScatter_{appendix}",
    #                    scatter_df = ctr_df, dst_type="Centroids", title=f"{repr+1} Centroids UMAP representation",
    #                    scatter_kwargs={"hue": "Data", "style": "Data"}, overwrite=overwrite)
    # raise Exception

    p_bar = pb.bar(total=(20))
    umap_study.scatter(umap_gb_df, output, hue="Data", dst_type="Good-Bad", title="Good-Bad UMAP representation",
                       appendix=f"{appendix}", p_bar=p_bar, overwrite=overwrite)
    umap_study.scatter(umap_df, output, hue="Data", appendix=f"{appendix}",
                       dst_type="All", title="All points UMAP representation",
                       legend_kwargs={"y": -0.37, "ncol": 3}, p_bar=p_bar, overwrite=overwrite)
    umap_study.kde(umap_gb_df, output, hue="Data", appendix=f"gb_{appendix}", p_bar=p_bar, overwrite=overwrite)
    umap_study.kde(umap_gb_df, output, hue="Data", appendix=f"Centroids_diffScatter_{appendix}",
                   p_bar=p_bar, scatter_df = umap_centroid_df, dst_type="Centroids",
                   scatter_kwargs={"hue": "Data", "style": "Data"}, overwrite=overwrite)
    correctness_order = ["TP", "FP", "TN", "FN", "U"]
    umap_study.kde(umap_gb_df, output, hue="Data", dst_type="Mahalanobis", overwrite=overwrite,
                   appendix=f"diffScatter_{appendix}", p_bar=p_bar,
                   scatter_df = umap_diff_df, scatter_kwargs={"hue": "M-Correctness",
                                                              "style": "M-Correctness",
                                                              "hue_order": correctness_order,
                                                              "style_order": correctness_order})
    umap_study.kde(umap_gb_df, output, hue="Data", appendix=f"diffScatter_{appendix}", p_bar=p_bar,
                   scatter_df = umap_diff_df, scatter_kwargs={"hue": "E-Correctness",
                                                              "style": "E-Correctness",
                                                              "hue_order": correctness_order,
                                                              "style_order": correctness_order},
                   overwrite=overwrite)
    umap_study.sub_kde(umap_gb_df, output, hue="Data", dst_type="Mahalanobis",
                       appendix=f"diffScatter_{appendix}", overwrite=overwrite,
                       scatter_df = umap_diff_df, sub_labels=[["TP"], ["FP"], ["TN"], ["FN"], ["U"]],
                       scatter_kwargs={"hue": "M-Correctness", "style": "M-Correctness",
                                       "hue_order": correctness_order, "style_order": correctness_order},
                       sub_column = "M-Correctness", p_bar=p_bar)
    umap_study.sub_kde(umap_gb_df, output, hue="Data", appendix=f"diffScatter_{appendix}",
                       scatter_df = umap_diff_df, sub_labels=[["TP"], ["FP"], ["TN"], ["FN"], ["U"]],
                       sub_column = "E-Correctness", p_bar=p_bar, overwrite=overwrite,
                       scatter_kwargs={"hue": "E-Correctness", "style": "E-Correctness",
                                       "hue_order": correctness_order, "style_order": correctness_order})
    umap_study.ecdf(full_diff_df[full_diff_df["M-Correctness"].isin(["TP", "FP"])], output,
                    x="d_m(P-cluster)", hue="M-Correctness", dst_type="Mahalanobis", overwrite=overwrite,
                    title="positive cluster distances ECDF", prefix="positive_cluster_dst",
                    legend_kwargs={"ncol": 2}, appendix=appendix, p_bar=p_bar)
    umap_study.ecdf(full_diff_df[full_diff_df["E-Correctness"].isin(["TP", "FP"])], output,
                    x="d_e(P-cluster)", hue="E-Correctness", overwrite=overwrite,
                    title="positive cluster distances ECDF", prefix="positive_cluster_dst",
                    legend_kwargs={"ncol": 2}, appendix=appendix, p_bar=p_bar)
    umap_study.ecdf(full_diff_df[full_diff_df["M-Correctness"].isin(["TN", "FN"])], output,
                    x="d_m(N-cluster)", hue="M-Correctness", dst_type="Mahalanobis", overwrite=overwrite,
                    title="positive cluster distances ECDF", prefix="positive_cluster_dst",
                    legend_kwargs={"ncol": 2}, appendix=appendix, p_bar=p_bar)
    umap_study.ecdf(full_diff_df[full_diff_df["E-Correctness"].isin(["TN", "FN"])], output,
                    x="d_e(N-cluster)", hue="E-Correctness", overwrite=overwrite,
                    title="negative cluster distances ECDF", prefix="negative_cluster_dst",
                    legend_kwargs={"ncol": 2}, appendix=appendix, p_bar=p_bar)
    p_bar.close()

    # x = np.arange(4)
    # p_bar = pb.bar(total=len(x))
    # for x, y in list(combinations(x, 2)):
    #     cm_sub_plot(full_gb_df, full_diff_df, ["TP", "TN"], x=f"X-{x}", y=f"X-{y}", cm_appendix=f"x{x}y{y}")
    #     cm_sub_plot(full_gb_df, full_diff_df, ["FP", "TN", "TP"], x=f"X-{x}", y=f"X-{y}", cm_appendix=f"x{x}y{y}")
    #     cm_sub_plot(full_gb_df, full_diff_df, ["FN", "TN", "TP"], x=f"X-{x}", y=f"X-{y}", cm_appendix=f"x{x}y{y}")
    #     cm_sub_plot(full_gb_df, full_centroid_df, ["p_centroids", "n_centroids"], x=f"X-{x}", y=f"X-{y}", cm_appendix=f"x{x}y{y}", hue="Data")
    #     p_bar.update(1)
    # p_bar.close()


if __name__ == "__main__":
    main()
