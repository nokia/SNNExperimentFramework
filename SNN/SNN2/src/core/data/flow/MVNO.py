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
class default flow
==================

Use this module to manage a default flow object.
"""

import time
import tensorflow as tf
import numpy as np
import pandas as pd

from typing import List, Optional, Dict, Any, Tuple

from SNN2.src.decorators.decorators import c_logger, cflow
from SNN2.src.core.data.flow.default import defaultFlow
from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.progressBar import pb
from SNN2.src.util.strings import s

@c_logger
class flow_op:
    """flow_op.
    Basic class for flow operations, this class is meant to be inherited by
    each op and extended with the correct behaviour.
    """

    def __init__(self,
                 *args,
                 Haction = None,
                 PklH = None,
                 **kwargs):
        """__init__.

        Parameters
        ----------
        """
        self.action_parm = Haction
        self.PklH = PklH

        if self.action_parm is None:
            raise Exception("The action handler is mandatory")

        self.steps = 0
        self.pkl_list = []

    def check_pkl(self) -> bool:
        """check_pkl.

        Parameters
        ----------
        """
        if self.PklH is None:
            self.write_msg("Pickle handler not available")
            return False
        return all([self.PklH.check(key) for key in self.pkl_list])

    def load(self) -> Dict[str, Any]:
        """load.

        Parameters
        ----------
        """
        if not self.check_pkl():
            return
        return {key: self.PklH.load(key) for key in self.pkl_list}

    def save(self, itms: List[Any]) -> None:
        """save.

        Parameters
        ----------
        """
        if not self.check_pkl():
            return
        for key, itm in zip(self.pkl_list, itms):
            self.write_msg(f"Saving {key}, {itm}")
            self.PklH.save(itm, key)


class MVNOFlow_load(flow_op):
    """MVNOFlow_load.
    Class used to manage and execute a specific sequence of operations for the
    loading of the dataframe.

    This class object receives the action_parm from the caller.
    The class object also receives the pkl handler.
    The method __call__ is used to execute the sequence of operations.

    It's possible to request the number of steps that this class object will
    exeucte through object.steps attribute.
    It's also avaialbe the attribue object.pkl_list that contains the list of
    pkl files that will be used in the sequence of operations.

    It available the method object.check_pkl that returns ture if the pkl files
    are available and internally loads the pkl files.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        """__init__.

        Parameters
        ----------
        """
        super().__init__(*args, **kwargs)

        self.steps = 8
        self.data = None
        self.pkl_list = ["data-rolled"]

    def local_load(self) -> None:
        """load.

        Parameters
        ----------
        """
        if not self.check_pkl():
            return
        self.data = self.load()["data-rolled"]

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        """__call__.

        Parameters
        ----------
        """
        self.write_msg("MVNO dataset loading operation")
        if self.data is not None:
            return self.data

        if self.check_pkl():
            self.local_load()
            return self.data

        self.data = self.action_parm.get_handler("load")(logger=self.logger)
        self.data = self.action_parm.get_handler("keepColumns")(df=self.data)
        self.data = self.action_parm.get_handler("ComputeRollFeatures")(df=self.data)
        self.data = self.action_parm.get_handler("ComputeLagFeatures")(df=self.data)
        self.data = self.action_parm.get_handler("ComputeDayOfWeek")(df=self.data)
        self.data = self.action_parm.get_handler("dropColumns")(df=self.data)
        self.data = self.action_parm.get_handler("dropNaN")(df=self.data)
        self.data = self.action_parm.get_handler("GroupStandardize")(df=self.data)
        self.write_msg(f"Trying to save data: {self.data}")
        self.save([self.data])
        return self.data


class MVNOFlow_getGoodBadDiff(flow_op):

    def __init__(self,
                 *args,
                 **kwargs):
        """__init__.

        Parameters
        ----------
        """
        super().__init__(*args, **kwargs)

        self.steps = 2
        self.good_prop = None
        self.bad_prop = None
        self.gray_prop = None
        self.pkl_list =["goods_properties", "bads_properties", "grays_properties"]

    def local_load(self) -> None:
        """load.

        Parameters
        ----------
        """
        if not self.check_pkl():
            return
        itms = self.load()
        self.good_prop = itms["goods_properties"]
        self.bad_prop = itms["bads_properties"]
        self.gray_prop = itms["grays_properties"]

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> List[DataManager]:
        """__call__.

        Parameters
        ----------
        """
        self.write_msg("MVNO Good Bad Grays separation operation")
        self.write_msg(f"Received df: \n{df}")
        # if all([elem is not None for elem in [self.good_prop, self.bad_prop, self.gray_prop]]):
        #     return [self.good_prop, self.bad_prop, self.gray_prop]
        #
        # if self.check_pkl():
        #     self.local_load()
        #     return self.data
        #
        # self.data = self.action_parm.get_handler("load")(logger=self.logger)
        # self.data = self.action_parm.get_handler("keepColumns")(df=self.data)
        # self.data = self.action_parm.get_handler("ComputeRollFeatures")(df=self.data)
        # self.data = self.action_parm.get_handler("ComputeLagFeatures")(df=self.data)
        # self.data = self.action_parm.get_handler("ComputeDayOfWeek")(df=self.data)
        # self.data = self.action_parm.get_handler("dropColumns")(df=self.data)
        # self.data = self.action_parm.get_handler("dropNaN")(df=self.data)
        # self.data = self.action_parm.get_handler("GroupStandardize")(df=self.data)
        # self.save([self.data])
        # return self.data


@cflow
class MVNOFlow(defaultFlow):
    """MVNO.

    PreProcessing flow for the MVNO dataset
	"""


    def __init__(self,
                 *args,
                 dataset: Optional[str] = None,
                 columns: Optional[List[str]] = None,
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
        self.gray_post_train_portion = gray_post_train_portion
        self.gray_in_train_portion = gray_in_train_portion
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

        self.data = None
        self.sep_prop = None # [self.goods_prop, self.bads_prop, self.grays_prop]

        self.flow = {
                "load": {
                        "handler": MVNOFlow_load,
                        "args": (None, ),
                        "kwargs": {"Haction": self.action_parm, "PklH": self.PklH},
                        "call_args": (None, ),
                        "call_kwargs": None,
                        "result_dst": self.data
                    }
                }
        # self.flow.update({
        #         "getGoodBadDiff": {
        #                 "handler": MVNOFlow_getGoodBadDiff,
        #                 "args": (None, ),
        #                 "kwargs": {"Haction": self.action_parm, "PklH": self.PklH},
        #                 "call_args": (self.data, ),
        #                 "call_kwargs": None,
        #                 "result_dst": self.sep_prop
        #             }
        #         })

        self.pbar = pb.bar(total=17)

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

        self.goods_prop, self.bads_prop, self.grays_prop = self.__get_good_bad_diff()
        self.goods_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            limit=5000, label="TfDataset")
        self.bads_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                           label="TfDataset")
        self.grays_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            label="TfDataset")

        self.write_msg(f"Goods keys: {list(self.goods_prop['Windows'].keys())}")

        del self.data

        self.gray_out_train, self.gray_in_train = self.__separate_difficult(self.grays_prop)

        self.gray_out_train.log_dump("Difficult out training normalized 0")
        self.training, self.validation, self.test = self.__get_trn_val_tst(
                    [self.goods_prop, self.bads_prop, self.gray_in_train],
                    pkl_l = [s.pkl_training, s.pkl_validation, s.pkl_test])
        del self.gray_in_train


        self.write_msg("training-validation-test separation done")

        self.gray_triplets = self.action_parm.get_handler("generatePredictionTriplets")(
                    self.gray_out_train, self.goods_prop, self.bads_prop,
                    keep_tf_dft=True, keep_anchor_wdw=True,
                    keep_all_wdw=True, logger=self.logger
                )

        # del self.goods_prop, self.bads_prop, self.gray_out_train
        # del self.goods_prop, self.bads_prop

        self.write_msg("Difficult Triplets generated")
        self.pbar.update(1)

        self.training_triplets = self.get_triplet(self.training, pkl_name="training_triplets")
        self.validation_triplets = self.get_triplet(self.validation, pkl_name="validation_triplets")
        self.test_triplets = self.get_triplet(self.test, pkl_name="test_triplets")

        self.write_msg("Triplets generated")

        self.gray_out_train.log_dump("Difficult out training normalized 1")

        self.goods_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            limit=5000, label="TfDataset")
        self.bads_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                           label="TfDataset")
        self.grays_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                            label="TfDataset")

        self.pbar.close()
        self.__state = "Terminated"

    def __apply_requests(self, df: pd.DataFrame, requests: DataManager) -> None:
        self.action_parm.get_handler("applyRequests")(
                df, window_clm="window",
                requests=requests,
                logger=self.logger
            )

    def __apply_window_separation(self, data: DataManager) -> DataManager:
        return self.action_parm.get_handler("windowsSeparationMVNO")(properties=data, logger=self.logger)

    def __apply_drop_outliers(self, data: DataManager) -> None:
        return self.action_parm.get_handler("windowDropOutliers")(data, logger=self.logger)

    def __get_good_bad_diff(self,
                            pkl_l: List[str] = ["goods_properties_s1", "bads_properties_s1", "grays_properties_s1"]) -> List[DataManager]:

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
