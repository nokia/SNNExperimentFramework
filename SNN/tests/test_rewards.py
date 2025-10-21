# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


from typing import Tuple, Union
import tensorflow as tf
import pytest

from SNN2.src.model.reward.original import original
from SNN2.src.model.reward.original import undecided
from SNN2.src.model.reward.accuracyMap import uRewardLimit, weightAccuracy

class TestRewards():


    @pytest.mark.parametrize(
            "CM,reward,tp_f,tn_f,fp_f,fn_f",
            [({"TP": 10, "TN": 10, "FP": 10, "FN": 10}, -100, 1,1,-10,-2)]
            )
    def test_original(self, CM,
                      reward,
                      tp_f, tn_f, fp_f, fn_f) -> None:
        r: int = original(CM, TP_factor=tp_f,
                          FP_factor=fp_f,
                          TN_factor=tn_f,
                          FN_factor=fn_f)
        assert r == reward

    @pytest.mark.parametrize(
            "CM,reward,tp_f,tn_f,fp_f,fn_f,u_f",
            [({"TP": 10, "TN": 10, "FP": 10, "FN": 10, "U": 10}, -150, 1, 1, -10, -2, -5)]
            )
    def test_undecided(self, CM,
                      reward,
                      tp_f, tn_f, fp_f, fn_f, u_f) -> None:
        r: int = undecided(CM, TP_factor=tp_f,
                           FP_factor=fp_f,
                           TN_factor=tn_f,
                           FN_factor=fn_f,
                           Undecided_factor=u_f)
        assert r == reward

    @pytest.mark.parametrize(
            "CM,reward,tp_f,tn_f,fp_f,fn_f,u_f",
            [({"TP": 10, "TN": 10, "FP": 10, "FN": 10, "U": 10}, -0.2, 1, 1, -1, -1, -1),
             ({"TP": 0, "TN": 0, "FP": 50, "FN": 50, "U": 0}, -1., 1, 1, -1, -1, -1),
             ({"TP": 50, "TN": 50, "FP": 0, "FN": 0, "U": 0}, 1., 1, 1, -1, -1, -1),
             ({"TP": 50, "TN": 0, "FP": 0, "FN": 0, "U": 50}, 0., 1, 1, -1, -1, -1),
             ({'TP': 2394, 'FP': 1628, 'TN': 827, 'FN': 385, 'U': 1286}, -0.012, 1, 1, -1, -1, -1)
             ]
            )
    def test_uRewardLimit(self, CM,
                          reward,
                          tp_f, tn_f, fp_f, fn_f, u_f) -> None:
        r: float = uRewardLimit(CM, TP_factor=tp_f,
                                FP_factor=fp_f,
                                TN_factor=tn_f,
                                FN_factor=fn_f,
                                Undecided_factor=u_f)
        assert r == reward

    @pytest.mark.parametrize(
            "CM,reward,tp_f,tn_f,fp_f,fn_f,u_f",
            [({"TP": 10, "TN": 10, "FP": 10, "FN": 10, "U": 10}, 0.4, 1, 1, 1, 1, 1),
             ({"TP": 0, "TN": 0, "FP": 50, "FN": 50, "U": 0}, 0.0, 1, 1, 1, 1, 1),
             ({"TP": 50, "TN": 50, "FP": 0, "FN": 0, "U": 0}, 1.0, 1, 1, 1, 1, 1),
             ({"TP": 42, "TN": 50, "FP": 0, "FN": 8, "U": 0}, 0.92, 1, 1, 1, 1, 1),
             ({"TP": 40, "TN": 50, "FP": 0, "FN": 10, "U": 0}, 0.90, 1, 1, 1, 1, 1),
             ({"TP": 30, "TN": 50, "FP": 0, "FN": 0, "U": 20}, 0.80, 1, 1, 1, 1, 1),
             ({"TP": 20, "TN": 50, "FP": 0, "FN": 0, "U": 30}, 0.70, 1, 1, 1, 1, 1),
             ({"TP": 0, "TN": 50, "FP": 0, "FN": 0, "U": 50}, 0.50, 1, 1, 1, 1, 1),
             ({"TP": 50, "TN": 0, "FP": 0, "FN": 0, "U": 50}, 0.50, 1, 1, 1, 1, 1),
             ({"TP": 0, "TN": 0, "FP": 50, "FN": 50, "U": 0}, 0.0, 1, 1, 2, 1, 1),
             ({"TP": 50, "TN": 50, "FP": 0, "FN": 0, "U": 0}, 1.0, 1, 1, 2, 1, 1),
             ({"TP": 50, "TN": 45, "FP": 5, "FN": 0, "U": 0}, 0.9048, 1, 1, 2, 1, 1),
             ({"TP": 50, "TN": 40, "FP": 10, "FN": 0, "U": 0}, 0.8182, 1, 1, 2, 1, 1),
             ({"TP": 50, "TN": 30, "FP": 20, "FN": 0, "U": 0}, 0.6667, 1, 1, 2, 1, 1),
             ({"TP": 50, "TN": 20, "FP": 30, "FN": 0, "U": 0}, 0.5385, 1, 1, 2, 1, 1),
             ({"TP": 50, "TN": 0, "FP": 50, "FN": 0, "U": 0}, 0.3333, 1, 1, 2, 1, 1),
             ]
            )
    def test_weightAccuracy(self, CM,
                            reward,
                            tp_f, tn_f, fp_f, fn_f, u_f) -> None:
        r = weightAccuracy(CM, TP_factor=tp_f,
                           FP_factor=fp_f,
                           TN_factor=tn_f,
                           FN_factor=fn_f,
                           Undecided_factor=u_f)
        print(r)
        assert r == reward

