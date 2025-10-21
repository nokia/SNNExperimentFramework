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

from typing import List, Optional, Dict, Any, Tuple
from SNN2.src.core.data.flow.default import defaultFlow
from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.io.logger import LogHandler as LH

from SNN2.src.io.progressBar import pb
from SNN2.src.decorators.decorators import c_logger, cflow


@cflow
class NetCatFlow(defaultFlow):
    """NetCatFlow.

    PreProcessing flow for the network categorical dataset
	"""


    def __init__(self,
                 *args,
                 dataset: Optional[str] = None,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        """__init__.

		Parameters
		----------
		"""
        assert columns is not None
        assert dataset is not None
        super().__init__(*args, **kwargs)

        self.columns = columns
        self.requests: Dict[str, Dict[str, Any]] = {
                        "Samples": {
                            'columns': self.columns,
                            'dtype': tf.float32,
                            'post_operation': None},
                        "Targets": {
                            'columns': ["web_service"],
                            'dtype': tf.string,
                            'post_operation': None},
                    }
        self.pbar = pb.bar(total=4)
        self.data = None

    def execute(self, *args, **kwargs) -> None:
        self.__state = "Executing"
        self.data = self.action_parm.get_handler("load")()
        self.pbar.update(1)

        self.data = self.action_parm.get_handler("dropColumns")(df=self.data)
        self.pbar.update(1)

        obj, self.targets_mapping, self.train_norm = self.__get_df_NFV_SDN(self.data, self.requests)
        self.pbar.update(1)
        training, validation, test = obj
        self.training, self.validation, self.test = self.__get_datasets_NFV_SDN(training, validation, test)
        self.pbar.update(1)
        self.pbar.close()
        self.__state = "Terminated"

    def __apply_kaggleDst_separation(self, df: pd.DataFrame, requests: DataManager) -> None:
        self.action_parm.get_handler("kaggleDstSeparation")(
                df,
                requests=requests,
                logger=self.logger
            )

    def __apply_TrnValTest_splitKaggle(self, data: DataManager) -> Tuple[DataManager,...]:
        return self.action_parm.get_handler("kaggleTrnValTestSeparation")(
                    data, logger=self.logger)

    def __apply_kaggleShuffle(self, data: DataManager) -> None:
        self.action_parm.get_handler("kaggleShuffle")(
                data, logger=self.logger)


    def __get_df_NFV_SDN(self,
                         data: pd.DataFrame,
                         requests: Dict[str, Dict[str, Any]],
                         data_manager_pkl_l: List[str] = ["training", "validation", "test"],
                         tensor_pkl_l: List[str] = ["TargetsMapping"]) -> Tuple[List[DataManager], tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
        if self.check_pkl_list(data_manager_pkl_l) and self.check_pkl_list(tensor_pkl_l):
            objs = self.load_pkl_list(data_manager_pkl_l, wrapper=to_dataManager)
            targets_mapping = self.load_pkl_list(tensor_pkl_l)

            train = objs[0]
            train_mean = self.action_parm.get_handler("featureMean")(train.dft("Samples"),logger=self.logger)
            train_std = self.action_parm.get_handler("featureStd")(train.dft("Samples"),logger=self.logger)

            self.pbar.update(2)
            return objs, targets_mapping, (train_mean, train_std)

        dataManager = DataManager(requests, logger=self.logger, default_field="tf_values")
        self.write_msg(f"Current data: {data}", level=LH.DEBUG)
        self.__apply_kaggleDst_separation(data, dataManager)
        dataManager.log_dump("GeneralDataManager")

        targets_mapping, targets_idx = tf.unique(tf.reshape(dataManager.dft("Targets"), [-1]))
        dataManager["Targets"].set_default(targets_idx)
        if not self.PklH.check('TargetsMapping'):
            self.save_pkl_dct({'TargetsMapping': targets_mapping})

        self.__apply_kaggleShuffle(dataManager)

        train, validation, test = self.__apply_TrnValTest_splitKaggle(dataManager)

        train_mean = self.action_parm.get_handler("featureMean")(train.dft("Samples"),logger=self.logger)
        train_std = self.action_parm.get_handler("featureStd")(train.dft("Samples"),logger=self.logger)

        self.write_msg(f"Train mean: {train_mean}")
        self.write_msg(f"Train std: {train_std}")

        if self.PklH.check(data_manager_pkl_l[0]):
            train = self.load_pkl_list([data_manager_pkl_l[0]], wrapper=to_dataManager)
            self.write_msg(f"Training loaded")
        else:
            train["Samples"]["tf_values"] = self.apply_normalize(train, train_mean, train_std, label="Samples")
            train.log_dump("AfterNormTrain")
            self.save_data([train], [data_manager_pkl_l[0]])

        if self.PklH.check(data_manager_pkl_l[1]):
            validation = self.load_pkl_list([data_manager_pkl_l[1]], wrapper=to_dataManager)
            self.write_msg(f"Validation loaded")
        else:
            validation["Samples"]["tf_values"] = self.apply_normalize(validation, train_mean, train_std, label="Samples")
            validation.log_dump("AfterNormVal")
            self.save_data([validation], [data_manager_pkl_l[1]])

        if self.PklH.check(data_manager_pkl_l[2]):
            test = self.load_pkl_list([data_manager_pkl_l[2]], wrapper=to_dataManager)
        else:
            test["Samples"]["tf_values"] = self.apply_normalize(test, train_mean, train_std, label="Samples")
            test.log_dump("AfterNormTest")
            self.save_data([test], [data_manager_pkl_l[2]])

        objs = [train, validation, test]
        return objs, targets_mapping, (train_mean, train_std)

    def __apply_catDataset_kaggle(self, data: DataManager) -> DataManager:
        triplet = self.action_parm.get_handler("generateKaggleCategorical")(data, logger=self.logger)
        return triplet

    def __get_datasets_NFV_SDN(self,
                               training: DataManager,
                               validation: DataManager,
                               test: DataManager) -> Tuple[DataManager, ...]:
        training = self.__apply_catDataset_kaggle(training)
        training.log_dump("Training")

        validation = self.__apply_catDataset_kaggle(validation)
        validation.log_dump("Validation")

        test = self.__apply_catDataset_kaggle(test)
        test.log_dump("Test")

        return training, validation, test
