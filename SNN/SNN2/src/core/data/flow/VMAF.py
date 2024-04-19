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
import pandas as pd
import numpy as np

from typing import List, Optional, Dict, Any, Tuple
from SNN2.src.contextManagers.contextManagers import timeit_cnt
from SNN2.src.core.data.flow.default import defaultFlow
from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.io.logger import LogHandler as LH

from SNN2.src.io.progressBar import pb
from SNN2.src.decorators.decorators import cflow

from SNN2.src.util.strings import s

@cflow
class VMAFFlow(defaultFlow):
    """VMAF.

    PreProcessing flow for the VMAF dataset
	"""


    def __init__(self,
                 *args,
                 dataset: Optional[str] = None,
                 columns: Optional[List[str]] = None,
                 window: int = 1,
                 vmaf_threshold: float = 80.0,
                 gray_post_train_portion: float = 0.7,
                 gray_in_train_portion: float = 0.3,
                 **kwargs):
        """__init__.

		Parameters
		----------
		"""
        assert columns is not None
        assert dataset is not None
        super().__init__(*args, **kwargs)

        self.columns = columns
        self.window = window
        self.vmaf_threshold = vmaf_threshold
        self.gray_post_train_portion = gray_post_train_portion
        self.gray_in_train_portion = gray_in_train_portion
        self.requests: Dict[str, Dict[str, Any]] = {
                    "Windows": {
                        'columns': self.columns,
                        'dtype': np.float32,
                        'post_operation': None},
                    "Targets": {
                        'columns': ["vmaf"],
                        'post_operation': [tf.math.reduce_mean, tf.reshape],
                        'post_operation_args': [(1, ), [-1]]},
                    "Video": {
                        'columns': ["video"],
                        'post_operation': [tf.gather, tf.reshape],
                        'post_operation_args': [(0, ), [-1]],
                        'post_operation_kwargs': [{'axis': 1}, {}]},
                    "Problem": {
                        'columns': ["problem"],
                        'post_operation': [tf.gather, tf.reshape],
                        'post_operation_args': [(0, ), [-1]],
                        'post_operation_kwargs': [{'axis': 1}, {}]},
                    "ProblemLevelApplied": {
                        'columns': ["value"],
                        'post_operation': [tf.gather, tf.reshape],
                        'post_operation_args': [(0, ), [-1]],
                        'post_operation_kwargs': [{'axis': 1}, {}]},
                    "OriginDataset": {
                        'columns': ["Dataset"],
                        'post_operation': [tf.gather, tf.reshape],
                        'post_operation_args': [(0, ), [-1]],
                        'post_operation_kwargs': [{'axis': 1}, {}]},
                    "ExpID": {
                        'columns': ["exp_id"],
                        'post_operation': [tf.gather, tf.reshape],
                        'post_operation_args': [(0, ), [-1]],
                        'post_operation_kwargs': [{'axis': 1}, {}]}
                }
        self.pbar = pb.bar(total=19)
        self.data = None

    def execute(self, *args, **kwargs) -> None:
        self.__state = "Executing"
        self.data = self.data = self.action_parm.get_handler("load")(logger=self.logger)
        self.pbar.update(1)
        self.data = self.action_parm.get_handler("dropOutliers")(df=self.data)
        self.pbar.update(1)
        self.data = self.action_parm.get_handler("dropOutliers")(df=self.data)
        self.pbar.update(1)
        self.data = self.action_parm.get_handler("GoodBadGraySeparation")(df=self.data)
        self.pbar.update(1)
        self.data = self.action_parm.get_handler("durationSeparation")(df=self.data)
        self.pbar.update(1)

        self.goods_prop, self.bads_prop, self.grays_prop = self.__get_good_bad_diff()
        self.pbar.update(1)
        self.goods_prop.log_dump("Goods")
        self.grays_prop.log_dump("Difficult")

        idx = tf.where(self.bads_prop.dft("Targets") > self.vmaf_threshold)[:, 0]
        self.write_msg(f"Bad samples with target higher than {self.vmaf_threshold}: {idx}")
        false_bads = self.bads_prop.sub_select(idx)
        false_bads.log_dump(f"False bads")
        self.write_msg(f"False bads original index: {false_bads.dft('ExpID')}")
        true_bads_idx = tf.where(self.bads_prop.dft("Targets") < self.vmaf_threshold)[:, 0].numpy()
        self.bads_prop = self.bads_prop.sub_select(true_bads_idx)
        self.write_msg(f"True bads:")
        self.bads_prop.log_dump("True bads")
        self.pbar.update(1)

        # print(f"Goods: {len(self.goods_prop.dft('Targets'))}")
        # print(f"Bads: {len(self.bads_prop.dft('Targets'))}")
        # print(f"Undecided: {len(self.grays_prop.dft('Targets'))}")
        # self.df: DataManager = MDF(["threshold_name","threshold_value","Statistic", "Value"])
        # def register(stat: str, value: Any) -> None:
        #     update = {"threshold_name": self.thr_name,
        #               "threshold_value": self.thr_value,
        #               "Statistic": stat,
        #               "Value": value}
        #     self.df.append(update)

        # register("G", len(self.goods_prop.dft('Targets')))
        # register("B", len(self.bads_prop.dft('Targets')))
        # register("U", len(self.grays_prop.dft('Targets')))
        # self.df.save(self.output_df)
        # raise Exception("Turn back on outliers droppout")
        self.gray_out_train, self.gray_in_train = self.__separate_difficult(self.grays_prop)

        self.gray_in_train.log_dump("Difficult in training")
        self.gray_out_train.log_dump("Difficult out training")
        self.pbar.update(1)

        self.gray_out_train.log_dump("Difficult out training normalized 0")
        self.training, self.validation, self.test, \
        self.gray_out_train, self.goods_prop, self.bads_prop, \
        self.grays_prop = self.__get_trn_val_tst(
                    [self.goods_prop, self.bads_prop, self.gray_in_train],
                    normalize_also=[self.gray_out_train, self.goods_prop, self.bads_prop, self.grays_prop],
                    pkl_l = [s.pkl_training, s.pkl_validation, s.pkl_test,
                             s.grays_wdw_norm, s.goods_wdw_norm, s.bads_wdw_norm,
                            "full_difficult_normalized"])

        self.gray_out_train.log_dump("Difficult out training normalized 1")

        self.training_triplets = self.get_triplet(self.training)
        self.validation_triplets = self.get_triplet(self.validation)
        self.test_triplets = self.get_triplet(self.test)
        self.test_triplets.log_dump("Test triplets")
        self.gray_triplets = self.action_parm.get_handler("generatePredictionTriplets")(
                    self.gray_out_train, self.goods_prop, self.bads_prop,
                    logger=self.logger
                )
        self.gray_triplets.log_dump("Difficult Triplets")
        self.pbar.update(1)

        # look for 473, 1738, 1994, 2017, 2919
        # look_for = [606, 1815, 2765, 2775]
        # look_for = [130, 433, 1236]
        # self.__analyze(look_for, stop=False)
        self.pbar.update(1)
        self.pbar.close()
        self.__state = "Terminated"

    def __apply_windowing(self, df: pd.DataFrame, requests: DataManager) -> None:
        self.action_parm.get_handler("listWindowing")(
                df, window=self.window,
                requests=requests,
                results_filed="tf_values",
                logger=self.logger
            )

    def __apply_window_separation(self, data: DataManager) -> DataManager:
        return self.action_parm.get_handler("windowsSeparation")(properties=data, logger=self.logger)

    def __apply_drop_outliers(self, data: DataManager) -> None:
        return self.action_parm.get_handler("windowDropOutliers")(data, logger=self.logger)

    def __get_good_bad_diff(self,
                            pkl_l: List[str] = ["goods_properties", "bads_properties", "grays_properties"]) -> List[DataManager]:

        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            objs = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            self.pbar.update(2)
            return objs

        dataManager = DataManager(self.requests, logger=self.logger, default_field="tf_values")
        self.write_msg(f"Current data: {self.data}", level=LH.DEBUG)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)
        self.__apply_windowing(self.data, dataManager)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)

        goods_prop, bads_prop, grays_prop = self.__apply_window_separation(dataManager)
        self.pbar.update(1)

        # drop goods_windows outliers
        goods_prop.log_dump("Goods")
        self.__apply_drop_outliers(goods_prop)
        self.pbar.update(1)

        goods_prop.log_dump("Goods")
        bads_prop.log_dump("Bads")
        grays_prop.log_dump("Difficult")

        objs = [goods_prop, bads_prop, grays_prop]
        self.save_data(objs, pkl_l)

        return objs

    def __separate_difficult(self, data: DataManager) -> Tuple[DataManager, ...]:
        gray_out_train, gray_in_train, _ = self.action_parm.get_handler("TrnValTstSeparation")(
                    data,
                    training_portion = self.gray_post_train_portion,
                    validation_portion = self.gray_in_train_portion,
                    test_portion=0.0,
                    to_dataset=False,
                    logger = self.logger
                )
        self.pbar.update(1)
        return gray_out_train, gray_in_train

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
        training.log_dump("Training")
        validation.log_dump("Validation")
        test.log_dump("Test")
        self.pbar.update(1)

        train_mean = self.action_parm.get_handler("featureMean")(training.dft("Windows"))
        train_std = self.action_parm.get_handler("featureStd")(training.dft("Windows"))
        self.train_norm = (train_mean, train_std)
        self.write_msg(f"Training mean: {train_mean}")
        self.write_msg(f"Training std: {train_std}")
        self.pbar.update(1)

        training.log_dump("TrainingPreNormalize")
        training["Windows"]["tf_values"] = self.apply_normalize(training, train_mean, train_std, label="Windows")
        training.log_dump("TrainingPostNormalize")
        validation["Windows"]["tf_values"] = self.apply_normalize(validation, train_mean, train_std, label="Windows")
        test["Windows"]["tf_values"] = self.apply_normalize(test, train_mean, train_std, label="Windows")

        for elem in normalize_also:
            elem["Windows"]["tf_values"] = self.apply_normalize(elem, train_mean, train_std, label="Windows")
        self.pbar.update(1)

        objs = [training, validation, test]
        objs.extend([elem for elem in normalize_also])
        self.save_data(objs, pkl_l)

        normalize_also[0].log_dump("Difficult normalized see here")
        objs[3].log_dump("Difficult normalized see here part 2")
        return objs

    def get_triplet(self, data: DataManager) -> DataManager:
        triplet = self.action_parm.get_handler("generateTripletsNG")(data, logger=self.logger)
        self.pbar.update(1)
        return triplet

