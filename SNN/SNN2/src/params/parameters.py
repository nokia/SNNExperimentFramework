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
class module
============

Use this module to manage all the parameters of a class.
"""

from __future__ import annotations

import ast
import numpy as np

from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH

from typing import Dict, Any, Callable, Optional

class _param:

    def none_log(self, *args, **kwargs):
        pass

    def __init__(self, d: Dict[str, str],
                 logger: Optional[LH] = None):
        self.name = d[s.io_name_key]
        self.type = d[s.io_type_key]
        self.value = d[s.param_value]
        self.logger = logger
        if self.logger is None:
            self.logger = self.none_log
        self.logger(self.__class__.__name__, f"{self.name} - {self.type} -> {self.value} Loaded")

    def __str__(self):
        return f"Param {self.name}, type: {self.type}, value: {self.value}"

class TypeSpecific_param(_param):
    def __init__(self, d: Dict[str, str],
                 param_type: str = s.param_generic_type,
                 **kwargs):
        if d[s.io_type_key] != param_type:
            raise Exception(f"Required {param_type} type, given {d[s.io_type_key]}")
        super().__init__(d, **kwargs)

class NumpyRng_param(TypeSpecific_param):

    def __init__(self, d: Dict[str, str], **kwargs):
        self.seed = int(d[s.param_value])
        d[s.param_value] = np.random.default_rng(self.seed)
        super().__init__(d, param_type=s.param_numpyRng_type, **kwargs)

class Callable_param(TypeSpecific_param):

    def __init__(self, d: Dict[str, Any],
                 function: Callable = None,
                 **kwargs):
        self.args = d[s.param_action_args]
        self.kwargs = d[s.param_action_kwargs]
        if function is None:
            raise Exception("The callable function must be provided!")
        self.function = function
        super().__init__(d, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        # print(*args)
        # print(*kwargs)
        self.kwargs.update(kwargs)
        return self.function(self.value, *args, *self.args, **self.kwargs)

    def __str__(self) -> str:
        input = super().__str__()
        return input + f" - args: {self.args} - kwargs: {self.kwargs}"
