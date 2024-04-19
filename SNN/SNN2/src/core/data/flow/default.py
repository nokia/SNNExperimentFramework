# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class default flow
==================

Use this module to manage a default flow object.
"""

from typing import Callable, List, Optional, Dict, Any, Tuple, Union

from SNN2.src.decorators.decorators import c_logger, cflow
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.core.data.DataManager import DataManager


@cflow
@c_logger
class defaultFlow:
    """defaultFlow.

	General flow handler class
	"""


    def __init__(self,
                 act_h,
                 *args,
                 PklH: Optional[PkH] = None,
                 **kwargs):
        """__init__.

		Parameters
		----------
		"""
        self.requests: Optional[Dict[str, Dict[str, Any]]] = None
        self.action_parm = act_h
        self.__state: str = "None"
        self.PklH = PklH
        pass

    def apply_normalize(self, data, mean, std, label: str = "windows") -> DataManager:
        return self.action_parm.get_handler("normalize")(
                data.dft(label),
                mean, std)

    def check_pkl_list(self, file_list: List[str]) -> Union[bool, None]:
        if self.PklH is None:
            return None
        return all([self.PklH.check(file) for file in file_list])

    def load_pkl_list(self, file_list: List[str], wrapper: Callable= None) -> Union[Tuple[Any, ...], None]:
        if self.PklH is None:
            return None

        if wrapper is None:
            def repeat(obj: Any):
                return obj
            wrapper = repeat
        return tuple([wrapper(self.PklH.load(file)) for file in file_list])

    def save_pkl_dct(self, obj_dct: Dict[str, Any], wrapper: Callable = None) -> None:
        if self.PklH is None:
            return None

        if wrapper is None:
            def repeat(obj: Any):
                return obj
            wrapper = repeat
        for key, obj in obj_dct.items():
            self.PklH.save(wrapper(obj), key)

    def save_data(self,
                  datasets: List[DataManager],
                  labels: List[str]) -> None:
        objs_dict = [elem.to_dict() for elem in datasets]
        assert len(objs_dict) == len(labels)
        save_dct = {key: obj for key, obj in zip(labels, objs_dict)}
        self.save_pkl_dct(save_dct)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"Execute must be implemented by the subclass")

    @property
    def state(self) -> str:
        return self.__state

