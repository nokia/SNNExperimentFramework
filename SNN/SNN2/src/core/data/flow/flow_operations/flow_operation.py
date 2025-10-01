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
            self.write_msg("Check pkl not passed")
            return
        for key, itm in zip(self.pkl_list, itms):
            self.write_msg(f"Saving {key}, {itm}")
            self.PklH.save(itm, key)

# class MVNOFlow_getGoodBadDiff(flow_op):
#
#     def __init__(self,
#                  *args,
#                  **kwargs):
#         """__init__.
#
#         Parameters
#         ----------
#         """
#         super().__init__(*args, **kwargs)
#
#         self.steps = 2
#         self.good_prop = None
#         self.bad_prop = None
#         self.gray_prop = None
#         self.pkl_list =["goods_properties", "bads_properties", "grays_properties"]
#
#     def local_load(self) -> None:
#         """load.
#
#         Parameters
#         ----------
#         """
#         if not self.check_pkl():
#             return
#         itms = self.load()
#         self.good_prop = itms["goods_properties"]
#         self.bad_prop = itms["bads_properties"]
#         self.gray_prop = itms["grays_properties"]
#
#     def __call__(self, df: pd.DataFrame, *args, **kwargs) -> List[DataManager]:
#         """__call__.
#
#         Parameters
#         ----------
#         """
#         self.write_msg("MVNO Good Bad Grays separation operation")
#         self.write_msg(f"Received df: \n{df}")
#         # if all([elem is not None for elem in [self.good_prop, self.bad_prop, self.gray_prop]]):
#         #     return [self.good_prop, self.bad_prop, self.gray_prop]
#         #
#         # if self.check_pkl():
#         #     self.local_load()
#         #     return self.data
#         #
#         # self.data = self.action_parm.get_handler("load")(logger=self.logger)
#         # self.data = self.action_parm.get_handler("keepColumns")(df=self.data)
#         # self.data = self.action_parm.get_handler("ComputeRollFeatures")(df=self.data)
#         # self.data = self.action_parm.get_handler("ComputeLagFeatures")(df=self.data)
#         # self.data = self.action_parm.get_handler("ComputeDayOfWeek")(df=self.data)
#         # self.data = self.action_parm.get_handler("dropColumns")(df=self.data)
#         # self.data = self.action_parm.get_handler("dropNaN")(df=self.data)
#         # self.data = self.action_parm.get_handler("GroupStandardize")(df=self.data)
#         # self.save([self.data])
#         # return self.data
#
#
