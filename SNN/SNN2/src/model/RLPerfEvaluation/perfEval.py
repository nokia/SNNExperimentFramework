# This program is free software: you can redistribute it and/or modify
# it under the terms of the graphNU grapheneral Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# graphNU grapheneral Public License for more details.
#
# You should have received a copy of the graphNU grapheneral Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2021 Mattia Milani <mattia.milani@nokia.com>

"""
Loss function Wrapper
=====================

Use this module to load a custom loss function

"""


from typing import Any
from SNN2.src.decorators.decorators import perfEval_functions

def selector(function, *args, **kwargs) -> Any:
    if function in perfEval_functions:
        return perfEval_functions[function](*args, **kwargs)
    else:
        raise ValueError(f"PerformanceEvaluation function \"{function}\" not available")

