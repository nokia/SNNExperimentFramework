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

import numpy as np
import tensorflow as tf

from SNN2.src.decorators.decorators import RLPerfEval, f_logger, reward

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

def convert(val: Any, input_type: Type, Transformer: Callable) -> Any:
    if isinstance(val, input_type):
        val = Transformer(val)
    return val

@RLPerfEval
@f_logger
def default(*args,
            **kwargs) -> bool:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    write_msg("Default performance evaluation function, alway returning False")
    return True

def get_correlation(a: np.ndarray, b: np.ndarray) -> float:
    correlation = np.corrcoef(a, b)
    return correlation[0, 1]

@f_logger
def get_margin_chainging_point(m: np.ndarray, p: np.ndarray, **kwargs) -> Union[float, None]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    tmp_p = np.round(p, 2)
    write_msg(f"rounded probabilities: {tmp_p}")
    res = np.where(tmp_p == 0.50)[0]
    write_msg(f"contact point: {res}")
    if len(res) == 0:
        return None
    return m[res[0]]


@RLPerfEval
@f_logger
def correlation(margin: tf.Tensor, probabilities: tf.Tensor,
                *args,
                current_episode: Optional[int] = None,
                p0_correlation_threshold: Union[str, float] = -90.0,
                p1_correlation_threshold: Union[str, float] = 90.0,
                margin_chainging_point_threshold: Union[str, float] = 0.0,
                exp_exploration_threshold: Union[str, int] = 6,
                **kwargs) -> bool:
    assert not current_episode is None

    p0_thr: float = convert(p0_correlation_threshold, str, float)
    p1_thr: float = convert(p1_correlation_threshold, str, float)
    m_thr: float = convert(margin_chainging_point_threshold, str, float)
    exp_thr: int = convert(exp_exploration_threshold, str, int)

    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    p0_m_correlation = get_correlation(margin.numpy(), probabilities[:, 0].numpy())
    p1_m_correlation = get_correlation(margin.numpy(), probabilities[:, 1].numpy())

    change_margin_point = get_margin_chainging_point(margin.numpy(), probabilities[:, 0].numpy(),
                                                     logger=logger)
    write_msg(f"Chainging margin point: {change_margin_point}")
    write_msg(f"Correlation values: p0-m {p0_m_correlation}, p1-m {p1_m_correlation}")
    print(f"Correlation values: p0-m {p0_m_correlation}, p1-m {p1_m_correlation}\nChainging m-value: {change_margin_point}")

    if p0_m_correlation < p0_thr and \
       p1_m_correlation > p1_thr and \
       current_episode >= exp_thr and \
       change_margin_point is not None and change_margin_point > m_thr:
           write_msg(f"Thresholds accomplished, no training needed anymore")
           print(f"Thresholds matched! {p0_m_correlation} - {p1_m_correlation}")
           return True
    return False
