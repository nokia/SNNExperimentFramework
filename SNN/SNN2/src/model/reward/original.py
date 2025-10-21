# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
