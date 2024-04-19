# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from SNN2.src.decorators.decorators import reward

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

def convert(val: Any, input_type: Type, Transformer: Callable) -> Any:
    if isinstance(val, input_type):
        val = Transformer(val)
    return val

def normalize(cm: Dict[str, int]) -> Dict[str, float]:
    total = sum(cm.values())
    return {k: v/total for k, v in cm.items()}


@reward
def uRewardLimit(confusion_matrix: Dict[str, int],
                 TP_factor: int = 1,
                 FP_factor: int = -1,
                 TN_factor: int = 1,
                 FN_factor: int = -1,
                 Undecided_factor: int = -1,
                 **kwargs) -> float:
    def normalize(cm):
        total = sum(cm.values())
        return {k: v/total for k, v in cm.items()}

    confusion_matrix: Dict[str, float] = normalize(confusion_matrix)

    tp: float = confusion_matrix["TP"]*TP_factor
    fp: float = confusion_matrix["FP"]*FP_factor
    tn: float = confusion_matrix["TN"]*TN_factor
    fn: float = confusion_matrix["FN"]*FN_factor
    u: float = confusion_matrix["U"]*Undecided_factor
    return round(sum([tp, fp, tn, fn, u]), 4)

@reward
def uRewardLimitPositive(confusion_matrix: Dict[str, int],
                         TP_factor: Union[int, str] = 1,
                         FP_factor: Union[int, str] = -2,
                         TN_factor: Union[int, str] = 1,
                         FN_factor: Union[int, str] = -1,
                         Undecided_factor: Union[int, str] = -1,
                         **kwargs) -> float:
    iTP_factor: int = convert(TP_factor, str, int)
    iFP_factor: int = convert(FP_factor, str, int)
    iTN_factor: int = convert(TN_factor, str, int)
    iFN_factor: int = convert(FN_factor, str, int)
    iUndecided_factor: int = convert(Undecided_factor, str, int)

    cm: Dict[str, float] = normalize(confusion_matrix)

    tp: float = cm["TP"]*iTP_factor
    fp: float = cm["FP"]*iFP_factor
    tn: float = cm["TN"]*iTN_factor
    fn: float = cm["FN"]*iFN_factor
    u: float = cm["U"]*iUndecided_factor
    return round(sum([tp, fp, tn, fn, u]), 4)+1.0

@reward
def weightAccuracy(confusion_matrix: Dict[str, int],
                   TP_factor: Union[int, str] = 1,
                   FP_factor: Union[int, str] = 2,
                   TN_factor: Union[int, str] = 1,
                   FN_factor: Union[int, str] = 1,
                   Undecided_factor: Union[int, str] = 1,
                   **kwargs) -> float:
    iTP_factor: int = convert(TP_factor, str, int)
    iFP_factor: int = convert(FP_factor, str, int)
    iTN_factor: int = convert(TN_factor, str, int)
    iFN_factor: int = convert(FN_factor, str, int)
    iUndecided_factor: int = convert(Undecided_factor, str, int)

    tp: float = confusion_matrix["TP"]*iTP_factor
    fp: float = confusion_matrix["FP"]*iFP_factor
    tn: float = confusion_matrix["TN"]*iTN_factor
    fn: float = confusion_matrix["FN"]*iFN_factor
    u: float = confusion_matrix["U"]*iUndecided_factor
    return round(sum([tp, tn])/sum([tp, fp, tn, fn, u]), 4)

def find_zone(value: float,
              zone_delimiter: Union[List[float], str] = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]) -> int:
    return np.where(np.array(zone_delimiter) <= value)[0][-1]

@reward
def zoneAcc(accuracy: float,
            zone_bonus: List[Union[int, float]] = [-1, 0.0, 0.2, 0.4, 0.8, 1.0, 2.0],
            **kwargs) -> Tuple[Union[int, float], float]:
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    new_zone = find_zone(accuracy, **kwargs)
    return zone_bonus[new_zone], accuracy

@reward
def zoneCM(confusion_matrix: Dict[str, int],
           **kwargs) -> Tuple[Union[int, float], float]:
    accuracy = weightAccuracy(confusion_matrix, **kwargs)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    return zoneAcc(accuracy)

@reward
def zoneParams(param: float,
               zone_delimiter: Union[List[float], str] = [0.0, 1.0, 3.0, 8.0],
               zone_bonus: List[Union[int, float]] = [0, 1, 2, 3],
               **kwargs) -> Tuple[Union[int, float], float]:
    param = 0.0 if param < 0.0 else param
    new_zone = find_zone(param, zone_delimiter=zone_delimiter)
    return zone_bonus[new_zone], param

@reward
def paramChange(confusion_matrix: Dict[str, int],
                current_params: float,
                previous_accuracy: Optional[float] = None,
                previous_params: Optional[float] = None,
                same_params_r: float = 1.0,
                improved_zone_r: float = 1.0,
                worst_zone_r: float = 0.2,
                otherwise_r: float = 0.8,
                **kwargs) -> Tuple[Union[int, float], float]:
    accuracy = weightAccuracy(confusion_matrix, **kwargs)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    previous_accuracy = previous_accuracy if previous_accuracy is not None else 0.0

    if current_params == previous_params:
        # print(f"Current and old parameters identical {current_params} - {previous_params}")
        return same_params_r, accuracy

    new_zone = zoneCM(confusion_matrix)[0]
    old_zone = zoneAcc(previous_accuracy)[0]

    if new_zone > old_zone:
        # print(f"New zone better than the previous one {new_zone} - {old_zone}")
        return improved_zone_r, accuracy
    elif new_zone < old_zone:
        # print(f"New zone worst than the previous one {new_zone} - {old_zone}")
        return worst_zone_r, accuracy

    return otherwise_r, accuracy

@reward
def cFunc(confusion_matrix: Dict[str, int],
          current_params: float,
          previous_accuracy: Optional[float] = None,
          previous_params: Optional[float] = None,
          best_r: float = 1.0,
          not_your_fault_r: float = 0.8,
          lucky_r: float = 0.5,
          bad_choice_r: float = 0.2,
          **kwargs) -> Tuple[Union[int, float], float]:
    accuracy = weightAccuracy(confusion_matrix, **kwargs)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    previous_accuracy = previous_accuracy if previous_accuracy is not None else 0.0
    new_zone = zoneCM(confusion_matrix)[0]
    old_zone = zoneAcc(previous_accuracy)[0]

    if new_zone > old_zone:
        return best_r, accuracy

    if current_params == previous_params:
        if new_zone == old_zone:
            return best_r, accuracy
        # print(f"Current and old parameters identical {current_params} - {previous_params}")
        return not_your_fault_r, accuracy

    if new_zone == old_zone:
        # print(f"New zone better than the previous one {new_zone} - {old_zone}")
        return lucky_r, accuracy
    return bad_choice_r, accuracy

@reward
def paramChangeEmu(confusion_matrix: Dict[str, int],
                   current_params: float,
                   previous_accuracy: Optional[float] = None,
                   previous_params: Optional[float] = None,
                   same_params_r: float = 1.0,
                   improved_zone_r: float = 1.0,
                   worst_zone_r: float = 0.25,
                   otherwise_r: float = 0.5,
                   **kwargs) -> Tuple[Union[int, float], float]:
    accuracy = weightAccuracy(confusion_matrix, **kwargs)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    previous_accuracy = previous_accuracy if previous_accuracy is not None else 0.0

    if current_params == previous_params:
        # print(f"Current and old parameters identical {current_params} - {previous_params}")
        return same_params_r, accuracy

    new_zone = zoneParams(current_params)[0]
    old_zone = zoneParams(previous_params)[0]

    if new_zone > old_zone:
        # print(f"New zone better than the previous one {new_zone} - {old_zone}")
        return improved_zone_r, accuracy
    elif new_zone < old_zone:
        # print(f"New zone worst than the previous one {new_zone} - {old_zone}")
        return worst_zone_r, accuracy

    return otherwise_r, accuracy

@reward
def zoneBlackHole(confusion_matrix: Dict[str, int],
                  current_params: float,
                  previous_accuracy: Optional[float] = None,
                  previous_params: Optional[float] = None,
                  **kwargs) -> Tuple[Union[int, float], float]:
    z = zoneCM(confusion_matrix, **kwargs)
    change = paramChange(confusion_matrix, current_params,
                         previous_accuracy=previous_accuracy,
                         previous_params=previous_params, **kwargs)

    z_r, c_r = z[0], change[0]
    # print(f"Zone: {z_r}, Change factor: {c_r}")
    return z_r*c_r, z[1]

@reward
def platoKeeper(confusion_matrix: Dict[str, int],
                current_params: float,
                previous_accuracy: Optional[float] = None,
                previous_params: Optional[float] = None,
                **kwargs) -> Tuple[Union[int, float], float]:
    z = zoneCM(confusion_matrix, **kwargs)
    change = cFunc(confusion_matrix, current_params,
                   previous_accuracy=previous_accuracy,
                   previous_params=previous_params, **kwargs)

    z_r, c_r = z[0], change[0]
    # print(f"Zone: {z_r}, Change factor: {c_r}")
    return z_r+c_r, z[1]

@reward
def fastBlackHole(confusion_matrix: Dict[str, int],
                  current_params: float,
                  previous_accuracy: Optional[float] = None,
                  previous_params: Optional[float] = None,
                  gamma: float = 0.99,
                  current_step: int = 0,
                  **kwargs) -> Tuple[Union[int, float], float]:
    z = zoneCM(confusion_matrix, **kwargs)
    change = paramChange(confusion_matrix, current_params,
                         previous_accuracy=previous_accuracy,
                         previous_params=previous_params,
                         improved_zone_r=1.5*(gamma**(2*current_step)),
                         otherwise_r=1.0*(gamma**(0.5*current_step)),
                         worst_zone_r=1.0*(gamma**current_step),
                         same_params_r=1.0,
                         **kwargs)
    z_r, c_r = z[0], change[0]
    return z_r*c_r, z[1]

@reward
def emulatedBlackHole(confusion_matrix: Dict[str, int],
                      current_params: float,
                      previous_accuracy: Optional[float] = None,
                      previous_params: Optional[float] = None,
                      gamma: float = 0.99,
                      current_step: int = 0,
                      **kwargs) -> Tuple[Union[int, float], float]:
    z = zoneParams(current_params, **kwargs)
    change = paramChangeEmu(confusion_matrix, current_params,
                            previous_accuracy=previous_accuracy,
                            previous_params=previous_params,
                            improved_zone_r=1.5*(gamma**(2*current_step)),
                            otherwise_r=1.0*(gamma**(0.5*current_step)),
                            worst_zone_r=1.0*(gamma**current_step),
                            same_params_r=1.0,
                            **kwargs)
    z_r, c_r = z[0], change[0]
    return z_r*c_r, z[1]

@reward
def zoneBonusMalus(confusion_matrix: Dict[str, int],
                   previous_accuracy: Optional[float] = None,
                   zone_delimiter: Union[List[float], str] = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95],
                   zone_bonus: Union[List[Union[int, float]], str] = [0, 1, 2, 3, 4, 5],
                   zone_malus: Union[List[Union[int, float]], str] = [-100, -100, -100, -100, -100, -100],
                   **kwargs) -> Tuple[float, float]:
    accuracy = round(uRewardLimitPositive(confusion_matrix=confusion_matrix, **kwargs)-1, 4)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    previous_accuracy = previous_accuracy if previous_accuracy is not None else 0.0

    new_zone = np.where(np.array(zone_delimiter) <= accuracy)[0][-1]
    old_zone = np.where(np.array(zone_delimiter) <= previous_accuracy)[0][-1]

    if new_zone > old_zone:
        return sum(zone_bonus[old_zone:new_zone]) + zone_bonus[new_zone], accuracy
    elif new_zone == old_zone:
        return zone_bonus[new_zone], accuracy
    return sum(zone_malus[new_zone:old_zone]), accuracy

@reward
def incdec(confusion_matrix: Dict[str, int],
           *args,
           previous_accuracy: Optional[float] = None,
           bonus: int = 1,
           neutral: int = 0,
           malus: int = 0,
           delta: float = 0.0,
           **kwargs) -> Tuple[int, float]:
    accuracy = round(weightAccuracy(confusion_matrix=confusion_matrix, **kwargs), 4)
    previous_accuracy = previous_accuracy if previous_accuracy is not None else 0.0

    if abs(accuracy-previous_accuracy) < delta:
        return neutral, accuracy

    if accuracy > previous_accuracy:
        return bonus, accuracy
    return malus, accuracy
