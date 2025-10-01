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

@cflow
class MVNOFlow_inference(defaultFlow):
    """MVNO.

    PreProcessing flow for the MVNO dataset
	"""


    def __init__(self,
                 *args,
                 dataset: Optional[str] = None,
                 columns: Optional[List[str]] = None,
                 gray_post_train_portion: float = 0.7,
                 gray_in_train_portion: float = 0.3,
                 inference_dst_name: str = "BADDIFF-infernece",
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
        self.dst_name = inference_dst_name
        # self.requests: Dict[str, Dict[str, Any]] = {
        #                 "Windows": {
        #                     'columns': self.columns,
        #                     'dtype': np.float16,
        #                     'post_operation': None},
        #                 "Targets": {
        #                     'columns': ["anomalous"],
        #                     'dtype': np.uint8,
        #                     'post_operation': [tf.math.reduce_sum, tf.reshape],
        #                     'post_operation_args': [(1, ), (-1, )]},
        #                 "ExpID": {
        #                     'columns': ["op_id", "timestamp"],
        #                     'dtype': np.int32,
        #                     'post_operation': [tf.gather],
        #                     'post_operation_args': [(0, )],
        #                     'post_operation_kwargs': [{'axis': 1}]}
        #             }
        self.requests: Dict[str, Dict[str, Any]] = {
                        "Windows": {
                            'columns': self.columns,
                            'dtype': np.float16,
                            'post_operation': None}
                    }

        self.data = None
        self.sep_prop = None # [self.goods_prop, self.bads_prop, self.grays_prop]

        self.pbar = pb.bar(total=9)

    def execute(self, *args, **kwargs) -> None:
        self.__state = "Executing"

        if not self.check_pkl_list([self.dst_name]):
            data = self.action_parm.get_handler("load")(logger=self.logger)
            data = self.action_parm.get_handler("keepColumns")(df=data)
            data = self.action_parm.get_handler("ComputeRollFeatures")(df=data)
            data = self.action_parm.get_handler("ComputeLagFeatures")(df=data)
            data = self.action_parm.get_handler("ComputeDayOfWeek")(df=data)
            data = self.action_parm.get_handler("dropColumns")(df=data)
            data = self.action_parm.get_handler("dropNaN")(df=data)
            data = self.action_parm.get_handler("GroupStandardize")(df=data)
            # data = self.action_parm.get_handler("DropAnomalous")(df=data)
            self.write_msg(f"Trying to save data: {data}")
            self.save_pkl_dct({self.dst_name: data})
            del data

        self.pbar.update(8)
        self.data = self.PklH.load(self.dst_name)
        # self.tmp_data = self.data[['op_id', 'timestamp', 'n_ul']].copy()
        # self.tmp_data['n_ul'] = self.tmp_data['n_ul'].astype(np.float16)
        # self.data.to_csv("BADDIFF_inference_retraining.csv", index=None)
        # self.tmp_data.to_csv("study_windows.csv", index=None)
        self.write_msg(f"{self.dst_name}: \n{self.data}")

        self.data_prop = self.__get_dstmanager(pkl_l=[f"{self.dst_name}_data_manager"])
        # self.write_msg(f"data_prop windows: {self.data_prop['Windows']['tf_values']}")
        # self.write_msg(f"data_prop windows op2 d1-start:\n{self.data_prop['Windows']['tf_values'][0, :, 0]}")
        # self.write_msg(f"data_prop windows op2 d1-end:\n{self.data_prop['Windows']['tf_values'][1081, :, 0]}")
        # self.write_msg(f"data_prop windows op2 d2-start:\n{self.data_prop['Windows']['tf_values'][1082, :, 0]}")
        # self.write_msg(f"data_prop windows op2 d2-end:\n{self.data_prop['Windows']['tf_values'][2521, :, 0]}")
        # self.write_msg(f"data_prop windows op2 d3-start:\n{self.data_prop['Windows']['tf_values'][2522, :, 0]}")
        # self.write_msg(f"data_prop windows op2 d3-end:\n{self.data_prop['Windows']['tf_values'][3961, :, 0]}")
        # self.write_msg(f"data_prop windows op2 d30-end:\n{self.data_prop['Windows']['tf_values'][42841, :, 0]}")
        # self.write_msg(f"data_prop windows op3 d1-start:\n{self.data_prop['Windows']['tf_values'][42842, :, 0]}")
        # self.write_msg(f"data_prop windows op3 d30-end:\n{self.data_prop['Windows']['tf_values'][85683, :, 0]}")
        # self.data_prop.transform_toDataset(inplace=True, keys=["Windows", "Targets", "ExpID"],
        #                                    label="TfDataset")
        self.data_prop.transform_toDataset(inplace=True, keys=["Windows"],
                                           label="TfDataset")

        self.data_prop["TripletDst"] = None
        self.data_prop["TripletDst"].set_default(
                    tf.stack([self.data_prop.dft("Windows"),
                              self.data_prop.dft("Windows"),
                              self.data_prop.dft("Windows")],
                             axis=1))
        self.data_prop["TripletDst"]["TfDataset"] = tf.data.Dataset.from_tensor_slices(
                    (self.data_prop.dft("Windows"),
                     self.data_prop.dft("Windows"),
                     self.data_prop.dft("Windows")))
        del self.data

        self.pbar.close()
        self.__state = "Terminated"

    def __apply_requests(self, df: pd.DataFrame, requests: DataManager) -> None:
        self.action_parm.get_handler("applyRequests")(
                df, window_clm="window",
                requests=requests,
                logger=self.logger
            )

    def __get_dstmanager(self,
                         pkl_l: List[str] = ["inference_dataset"]) -> DataManager:

        if self.check_pkl_list(pkl_l):
            to_dataManager = lambda x: DataManager(x, default_field="tf_values", logger=self.logger)
            objs = self.load_pkl_list(pkl_l, wrapper=to_dataManager)
            if len(objs) > 1:
                raise Exception("Too many objects loaded")
            self.pbar.update(1)
            return objs[0]

        dataManager = DataManager(self.requests, logger=self.logger, default_field="tf_values")
        self.write_msg(f"Current data: {self.data}", level=LH.DEBUG)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)
        self.__apply_requests(self.data, dataManager)
        self.write_msg(f"Current dataManager: {dataManager}", level=LH.DEBUG)

        self.pbar.update(1)
        dataManager.log_dump("Inference data")
        self.save_data([dataManager], pkl_l)

        return dataManager
