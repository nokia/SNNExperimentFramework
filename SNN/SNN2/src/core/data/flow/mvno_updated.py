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
class MVNO flow
===============

This class uses a smarter memory management system to avoid memory issues.
The common philosophy of this module is lazy evaluation and work on the
least ammount of information possible before generating the next step.
"""

from typing import List, Optional, Dict, Any

import tensorflow as tf
import numpy as np

from SNN2.src.decorators.decorators import cflow
from SNN2.src.core.data.flow.default import defaultFlow
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.progressBar import pb
from SNN2.src.util.strings import s

from .flow_operations import mvno_load

@cflow
class MVNOFlow_smartMemory(defaultFlow):
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

        self.flow = [
                    mvno_load(Haction=self.action_parm,
                              PklH=self.PklH,
                              logger = self.logger)
                ]

        self.pbar = pb.bar(total=sum([op.steps for op in self.flow]))

    def execute(self, *args, **kwargs) -> None:
        self.__state = "Executing"

        self.write_msg(f"MVNO-smart memory flow execution")

        for operation in self.flow:
            self.write_msg(f"Operation: {operation.name}")
            self.data = operation()
            self.write_msg(f"Object: {operation}")

        # if not self.check_pkl_list(["data_rolled"]):
        #     data = self.action_parm.get_handler("load")(logger=self.logger)
        #     data = self.action_parm.get_handler("keepColumns")(df=data)
        #     data = self.action_parm.get_handler("ComputeRollFeatures")(df=data)
        #     data = self.action_parm.get_handler("ComputeLagFeatures")(df=data)
        #     data = self.action_parm.get_handler("ComputeDayOfWeek")(df=data)
        #     data = self.action_parm.get_handler("dropColumns")(df=data)
        #     data = self.action_parm.get_handler("dropNaN")(df=data)
        #     data = self.action_parm.get_handler("GroupStandardize")(df=data)
        #     self.write_msg(f"Trying to save data: {data}")
        #     self.save_pkl_dct({'data_rolled': data})
        #     del data

        self.pbar.close()
        self.__state = "Terminated"
