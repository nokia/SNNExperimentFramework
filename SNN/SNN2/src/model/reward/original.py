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

import tensorflow as tf

from SNN2.src.decorators.decorators import reward

from typing import Dict

@reward
def original(confusion_matrix: Dict[str, int],
             TP_factor: int = 1,
             FP_factor: int = -10,
             TN_factor: int = 1,
             FN_factor: int = -2,
            **kwargs) -> int:
    tp:int = confusion_matrix["TP"]*TP_factor
    fp:int = confusion_matrix["FP"]*FP_factor
    tn:int = confusion_matrix["TN"]*TN_factor
    fn:int = confusion_matrix["FN"]*FN_factor
    return sum([tp, fp, tn, fn])

@reward
def undecided(confusion_matrix: Dict[str, int],
              TP_factor: int = 1,
              FP_factor: int = -10,
              TN_factor: int = 1,
              FN_factor: int = -2,
              Undecided_factor: int = -5,
              **kwargs) -> int:
    tp:int = confusion_matrix["TP"]*TP_factor
    fp:int = confusion_matrix["FP"]*FP_factor
    tn:int = confusion_matrix["TN"]*TN_factor
    fn:int = confusion_matrix["FN"]*FN_factor
    u:int = confusion_matrix["U"]*Undecided_factor
    return sum([tp, fp, tn, fn, u])
