# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class default flow
==================

Use this module to manage a default flow object.
"""

import time
import tensorflow as tf
import numpy as np
import pandas as pd

from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy

from SNN2.src.decorators.decorators import c_logger, cflow
from SNN2.src.core.data.flow.default import defaultFlow
from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.progressBar import pb
from SNN2.src.util.strings import s

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32)))
def pp_dst(points: tf.Tensor, adv: tf.Tensor) -> tf.Tensor:
    points_shape = tf.shape(points)
    adv_shape = tf.shape(adv)
    points_r = tf.repeat(points, repeats=adv_shape[0], axis=0)
    adv_r = tf.tile(adv, multiples=[tf.shape(points_r)[0] // adv_shape[0], 1])
    # all_dst = tf.math.square(tf.norm(tf.math.subtract(points_r, adv_r), axis=-1))
    all_dst = tf.norm(tf.math.subtract(points_r, adv_r), axis=-1,
    ord='euclidean')
    all_dst = tf.reshape(all_dst, (points_shape[0], adv_shape[0]))
    all_dst = tf.transpose(all_dst)
    return all_dst

@cflow
class MVNOFlow_extended(defaultFlow):
    """MVNO.

    PreProcessing flow for the MVNO dataset
	"""


    def __init__(self,
                 *args,
                 dataset: Optional[str] = None,
                 columns: Optional[List[str]] = None,
                 gray_post_train_portion: float = 0.7,
                 gray_in_train_portion: float = 0.3,
                 extra_dataset_name: str = "extra_dataset",
                 extra_dataset_base_name: str = "extra_dataset_base",
                 use_random_samples: bool = False,
                 **kwargs):
        """__init__.

		Parameters
		----------
		"""
        assert columns is not None
        assert dataset is not None
        super().__init__(*args, **kwargs)

        self.columns = columns
        self.gray_post_train_portion = gray_post_train_portion
        self.gray_in_train_portion = gray_in_train_portion
        self.extra_dataset_name = extra_dataset_name
        self.extra_dataset_base_name = extra_dataset_base_name
        self.use_random_samples = use_random_samples
        self.requests: Dict[str, Dict[str, Any]] = {
                        "Windows": {
                            'columns': self.columns,
                            'dtype': np.float16,
                            'post_operation': None},
                        "Targets": {
                            'columns': ["anomalous"],
                            'dtype': np.uint8,
                            'post_operation': [tf.math.reduce_sum, tf.reshape],
                            'post_operation_args': [(1, ), (-1, )]},
                        "ExpID": {
                            'columns': ["op_id", "timestamp"],
                            'dtype': np.int32,
                            'post_operation': [tf.gather],
                            'post_operation_args': [(0, )],
                            'post_operation_kwargs': [{'axis': 1}]}
                    }
        self.extra_requests: Dict[str, Dict[str, Any]] = {
                        "Windows": {
                            'columns': self.columns,
                            'dtype': np.float16,
                            'post_operation': None},
                        "Targets": {
                            'columns': ["anomalous"],
                            'dtype': np.uint8,
                            'post_operation': [tf.math.reduce_sum, tf.reshape],
                            'post_operation_args': [(1, ), (-1, )]},
                        "ExpID": {
                            'columns': ["op_id", "timestamp"],
                            'dtype': np.int32,
                            'post_operation': [tf.gather],
                            'post_operation_args': [(0, )],
                            'post_operation_kwargs': [{'axis': 1}]}
                    }

        self.data = None
        self.sep_prop = None # [self.goods_prop, self.bads_prop, self.grays_prop]

        self.pbar = pb.bar(total=20)

    def execute(self, *args, **kwargs) -> None:
        self.__state = "Executing"

        if not self.check_pkl_list(["data_rolled"]):
            data = self.action_parm.get_handler("load")(logger=self.logger)
            data = self.action_parm.get_handler("keepColumns")(df=data)
            data = self.action_parm.get_handler("ComputeRollFeatures")(df=data)
            data = self.action_parm.get_handler("ComputeLagFeatures")(df=data)
            data = self.action_parm.get_handler("ComputeDayOfWeek")(df=data)
            data = self.action_parm.get_handler("dropColumns")(df=data)
            data = self.action_parm.get_handler("dropNaN")(df=data)
            data = self.action_parm.get_handler("GroupStandardize")(df=data)
            self.write_msg(f"Trying to save data: {data}")
            self.save_pkl_dct({'data_rolled': data})
            del data

        self.pbar.update(8)
        self.data = self.PklH.load("data_rolled")
        self.write_msg(f"Data loaded: \n{self.data}")

        self.write_msg("Loading the extra dataset for the training!")

        if not self.check_pkl_list([self.extra_dataset_base_name]):
            extra_data = self.action_parm.get_handler("load_extra")(logger=self.logger)
            extra_data = self.action_parm.get_handler("ComputeRollFeatures")(df=extra_data)
            extra_data = self.action_parm.get_handler("ComputeLagFeatures")(df=extra_data)
            extra_data = self.action_parm.get_handler("ComputeDayOfWeek")(df=extra_data)
            extra_data = self.action_parm.get_handler("dropColumns")(df=extra_data)
            extra_data = self.action_parm.get_handler("dropNaN")(df=extra_data)
            extra_data = self.action_parm.get_handler("GroupStandardize")(df=extra_data)
            self.write_msg(f"Trying to save extra extra_data: {extra_data}")
            self.save_pkl_dct({f"{self.extra_dataset_base_name}": extra_data})
            del extra_data

        self.extra_data = self.PklH.load(self.extra_dataset_base_name)
        self.write_msg(f"Extra Data loaded: \n{self.extra_data}")

        if not self.check_pkl_list([self.extra_dataset_name]):
            self.extra_data = self.action_parm.get_handler("DropNotAnomalous")(df=self.extra_data)
            self.write_msg(f"Trying to save extra extra_data: {self.extra_data}")
            self.save_pkl_dct({f"{self.extra_dataset_name}": self.extra_data})
            del self.extra_data

        self.extra_data = self.PklH.load(self.extra_dataset_name)
        self.write_msg(f"Extra Data loaded: \n{self.extra_data}")

        self.goods_prop, self.bads_prop, self.grays_prop = self.__get_good_bad_diff()
        self.write_msg(f"Goods keys: {list(self.goods_prop['Windows'].keys())}")
        del self.data

        self.extra_goods_prop, self.extra_bads_prop, _ = self.__get_extra_dst(
                pkl_l = [f"{self.extra_dataset_name}_goods_properties",
                         f"{self.extra_dataset_name}_bads_properties",
                         f"{self.extra_dataset_name}_grays_properties"])
        del self.extra_data

        self.gray_out_train, self.gray_in_train = self.__separate_difficult(self.grays_prop)
        self.extra_bad_training = self.__extra_separate(self.extra_bads_prop,
                                                        pkl_l = [f"{self.extra_dataset_name}_bads_separate"])[0]
        self.extra_good_training = self.__extra_separate(self.extra_goods_prop,
                                                        pkl_l = [f"{self.extra_dataset_name}_goods_separate"])[0]

        self.gray_out_train.log_dump("Difficult out training normalized 0")
        self.training, self.validation, self.test = self.__get_trn_val_tst(
                    [self.goods_prop, self.bads_prop, self.gray_in_train],
                    pkl_l = [s.pkl_training, s.pkl_validation, s.pkl_test])
        del self.gray_in_train

        if not self.check_pkl_list([f"{self.extra_dataset_name}_representatives_good",
                                    f"{self.extra_dataset_name}_representatives_bad"]):
            g_emb = self.action_parm.get_handler("load_good_emb")(pkl=self.PklH)
            b_emb = self.action_parm.get_handler("load_bad_emb")(pkl=self.PklH)
            g_centr = self.action_parm.get_handler("compute_centroids")(g_emb)
            b_centr = self.action_parm.get_handler("compute_centroids")(b_emb)
            g_samples = self.action_parm.get_handler("load_good_samples")(pkl=self.PklH)
            b_samples = self.action_parm.get_handler("load_bad_samples")(pkl=self.PklH)
            # Identify the closest sample to the centroid of each group
            # Identify the closest emb and then use the same index to identify the sample
            g_dst = tf.squeeze(pp_dst(g_emb, g_centr))
            b_dst = tf.squeeze(pp_dst(b_emb, b_centr))

            min_g_dst_idx = tf.math.argmin(g_dst, axis=0)
            min_b_dst_idx = tf.math.argmin(b_dst, axis=0)

            g_representative = tf.gather(g_samples, min_g_dst_idx)
            b_representative = tf.gather(b_samples, min_b_dst_idx)
            self.save_pkl_dct({f"{self.extra_dataset_name}_representatives_good": g_representative,
                               f"{self.extra_dataset_name}_representatives_bad": b_representative})

        g_representative = self.PklH.load(f"{self.extra_dataset_name}_representatives_good")
        b_representative = self.PklH.load(f"{self.extra_dataset_name}_representatives_bad")

        self.write_msg("training-validation-test separation done")

        self.gray_triplets = self.action_parm.get_handler("generatePredictionTriplets")(
                    self.gray_out_train, self.goods_prop, self.bads_prop,
                    keep_tf_dft=True, keep_anchor_wdw=True,
                    keep_all_wdw=True, logger=self.logger
                )

        self.write_msg("Difficult Triplets generated")
        self.pbar.update(1)

        if self.use_random_samples:
            # Investigate usage of GenerateGrayTriplets function!
            self.extra_bad_triplets = self.action_parm.get_handler("GenerateGrayTriplets")(
                    self.extra_bad_training, self.bads_prop, self.good_prop,
                    logger=self.logger, keep_all=True)
            self.extra_good_triplets = self.action_parm.get_handler("GenerateGrayTriplets")(
                    self.extra_good_training, self.goods_prop, self.bads_prop,
                    logger=self.logger, keep_all=True)
        else:
            extra_b_shape = self.extra_bad_training.dft("Windows").shape
            extra_g_shape = self.extra_good_training.dft("Windows").shape
            g_rep_p = tf.repeat(tf.expand_dims(g_representative, axis=0), repeats=extra_g_shape[0], axis=0)
            g_rep_n = tf.repeat(tf.expand_dims(b_representative, axis=0), repeats=extra_g_shape[0], axis=0)
            b_rep_p = tf.repeat(tf.expand_dims(b_representative, axis=0), repeats=extra_b_shape[0], axis=0)
            b_rep_n = tf.repeat(tf.expand_dims(g_representative, axis=0), repeats=extra_b_shape[0], axis=0)

            self.extra_bad_triplets = deepcopy(self.extra_bad_training)
            self.extra_good_triplets = deepcopy(self.extra_good_training)
            self.extra_bad_triplets["TripletDst"] = None
            self.extra_bad_triplets["TripletDst"].set_default(
                                    tf.stack([b_rep_p,
                                              self.extra_bad_triplets.dft("Windows"),
                                              b_rep_n],
                                              axis=1))
            self.extra_bad_triplets["TripletDst"]["TfDataset"] = \
                    tf.data.Dataset.from_tensor_slices(
                        (b_rep_p,
                         self.extra_bad_triplets.dft("Windows"),
                         b_rep_n))

            self.extra_good_triplets["TripletDst"] = None
            self.extra_good_triplets["TripletDst"].set_default(
                                    tf.stack([g_rep_p,
                                              self.extra_good_triplets.dft("Windows"),
                                              g_rep_n],
                                              axis=1))
            self.extra_good_triplets["TripletDst"]["TfDataset"] = \
                    tf.data.Dataset.from_tensor_slices(
                        (g_rep_p,
                         self.extra_good_triplets.dft("Windows"),
                         g_rep_n))

        self.training_triplets = self.get_triplet(self.training, pkl_name="training_triplets")
        self.validation_triplets = self.get_triplet(self.validation, pkl_name="validation_triplets")
        self.test_triplets = self.get_triplet(self.test, pkl_name="test_triplets")

        self.training_triplets["TripletDst"]["TfDataset"] = self.__merge_triplets(self.extra_bad_triplets,
                                                       self.training_triplets)
        self.training_triplets["TripletDst"]["TfDataset"] = self.__merge_triplets(self.extra_good_triplets,
                                                       self.training_triplets)

        self.write_msg("Triplets generated")

        self.extra_bad_training.log_dump("Extra bad training")
        self.training_triplets.log_dump("Training triplets")

        self.goods_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            limit=5000, label="TfDataset")
        self.bads_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                           label="TfDataset")
        self.grays_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            label="TfDataset")
        self.extra_bads_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            limit=5000, label="TfDataset")
        self.extra_goods_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            limit=5000, label="TfDataset")

        self.pbar.close()
        self.__state = "Terminated"

    def __merge_triplets(self,
                         new_data: DataManager,
                       old_data: DataManager,
                       *args,
                       batch_size: Optional[int] = None,
                       **kwargs) -> tf.data.Dataset:
        # new_data.log_dump("New Data")
        # old_data.log_dump("Old Data")

        new_dst = new_data["TripletDst"]["TfDataset"]
        old_dst = old_data["TripletDst"]["TfDataset"]

        new_dst = new_dst.concatenate(old_dst)
        self.write_msg(f"Dataset merged: {new_dst.cardinality().numpy()}")
        self.write_msg(f"Dataset merged: {new_dst}")

        new_dst = new_dst.shuffle(50000)
        if batch_size is not None:
            new_dst = new_dst.batch(batch_size)

        self.write_msg(f"Dataset merged: {new_dst.cardinality().numpy()}")
        self.write_msg(f"Dataset merged: {new_dst}")
        # write_msg(f"Dataset merged: {new_dst.cardinality().numpy()}")
        return new_dst

    def __apply_requests(self, df: pd.DataFrame, requests: DataManager) -> None:
        self.action_parm.get_handler("applyRequests")(
                df, window_clm="window",
                requests=requests,
                logger=self.logger
            )

    def __apply_extra_requests(self, df: pd.DataFrame, requests: DataManager) -> None:
        self.action_parm.get_handler("extra_applyRequests")(
                df, window_clm="window",
                requests=requests,
                logger=self.logger
            )

    def __apply_window_separation(self, data: DataManager) -> DataManager:
        return self.action_parm.get_handler("windowsSeparationMVNO")(properties=data, logger=self.logger)

    def __apply_drop_outliers(self, data: DataManager) -> None:
        return self.action_parm.get_handler("windowDropOutliers")(data, logger=self.logger)

    def __get_good_bad_diff(self,
                            pkl_l: List[str] = ["goods_properties", "bads_properties", "grays_properties"]) -> List[DataManager]:

        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            objs = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            self.pbar.update(1)
            return objs

        dataManager = DataManager(self.requests, logger=self.logger, default_field="tf_values")
        self.write_msg(f"Current data: {self.data}", level=LH.DEBUG)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)
        self.__apply_requests(self.data, dataManager)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)

        goods_prop, bads_prop, grays_prop = self.__apply_window_separation(dataManager)
        self.pbar.update(1)

        # drop goods_windows outliers
        goods_prop.log_dump("Goods")
        bads_prop.log_dump("Bads")
        grays_prop.log_dump("Difficult")

        objs = [goods_prop, bads_prop, grays_prop]
        self.save_data(objs, pkl_l)

        return objs

    def __get_extra_dst(self,
                        pkl_l: str = ["extra_goods_properties",
                                      "extra_bads_properties",
                                      "extra_grays_properties"]) -> List[DataManager]:

        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            objs = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            self.pbar.update(1)
            return objs

        dataManager = DataManager(self.extra_requests, logger=self.logger, default_field="tf_values")
        self.write_msg(f"Current data: {self.extra_data}", level=LH.DEBUG)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)
        self.__apply_extra_requests(self.extra_data, dataManager)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)

        goods_prop, bads_prop, grays_prop = self.__apply_window_separation(dataManager)
        self.pbar.update(1)

        # drop goods_windows outliers
        goods_prop.log_dump("Goods")
        bads_prop.log_dump("Bads")
        grays_prop.log_dump("Difficult")

        objs = [goods_prop, bads_prop, grays_prop]
        self.save_data(objs, pkl_l)

        return objs

    def __separate_difficult(self, data: DataManager,
                             pkl_l: List[str] = ["separation-grays_out_train", "separation-grays_in_train"]) -> Tuple[DataManager, ...]:
        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            objs = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            self.pbar.update(1)
            return objs


        gray_out_train, gray_in_train, _ = self.action_parm.get_handler("TrnValTstSeparation")(
                    data,
                    training_portion = self.gray_post_train_portion,
                    validation_portion = self.gray_in_train_portion,
                    test_portion=0.0,
                    to_dataset=False,
                    logger = self.logger
                )
        self.pbar.update(1)

        self.save_data([gray_out_train, gray_in_train], pkl_l)
        return gray_out_train, gray_in_train

    def __extra_separate(self, data: DataManager,
                             pkl_l: List[str] = ["separation-extra_bad"]) -> Tuple[DataManager, ...]:
        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            objs = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            self.pbar.update(1)
            return objs


        extra_bad_training, _, _ = self.action_parm.get_handler("TrnValTstSeparation")(
                    data,
                    training_portion = 1.0,
                    validation_portion = 0.0,
                    test_portion=0.0,
                    to_dataset=False,
                    logger = self.logger
                )
        self.pbar.update(1)

        self.save_data([extra_bad_training], pkl_l)
        return [extra_bad_training]

    def __get_trn_val_tst(self,
                          datasets: List[DataManager],
                          pkl_l: List[str] = [s.pkl_training,
                                              s.pkl_validation,
                                              s.pkl_test,
                                              s.grays_wdw_norm,
                                              s.goods_wdw_norm,
                                              s.bads_wdw_norm,
                                              "full_difficult_normalized"],
                          normalize_also: List[DataManager] = []) -> Tuple[DataManager, ...]:
        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            obj = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            self.pbar.update(3)
            return obj

        # Genrate the first training-validation-test datasets
        training, validation, test = self.action_parm.get_handler("BalanceSeparationNG")(datasets, logger=self.logger)
        self.pbar.update(1)

        objs = [training, validation, test]
        self.save_data(objs, pkl_l)

        return objs

    def get_triplet(self, data: DataManager,
                    pkl_name: str = "default_get_triplet_name") -> DataManager:
        # if self.check_pkl_list([pkl_name]):
        #     to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
        #     obj = self.load_pkl_list([pkl_name], wrapper=to_dataManager)
        #     self.pbar.update(1)
        #     return obj

        triplet = self.action_parm.get_handler("generateTripletsNG")(data,
            keep_all=True, logger=self.logger)
        self.pbar.update(1)

        # self.save_data([triplet], [pkl_name])
        return triplet
