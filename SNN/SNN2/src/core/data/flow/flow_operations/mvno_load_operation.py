# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class default flow
==================

Use this module to manage a default flow object.
"""

from typing import List, Optional, Dict, Any, Tuple

import time
import tensorflow as tf
import numpy as np
import pandas as pd

from SNN2.src.decorators.decorators import c_logger, cflow
from SNN2.src.core.data.flow.default import defaultFlow
from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.progressBar import pb
from SNN2.src.util.strings import s

from .flow_operation import flow_op

class mvno_load(flow_op):
    """MVNO_load.
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

        self.name = "mvno_load_data"
        self.steps = 8
        self.data = None
        self.pkl_list = ["data-rolled"]

    def local_load(self) -> None:
        """load.

        Parameters
        ----------
        """
        if not self.check_pkl():
            return None
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



