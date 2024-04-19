# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


import tensorflow as tf
import numpy as np
import pytest

from SNN2.src.environment.conf import set_rngs
from SNN2.src.util.strings import s
from SNN2.src.params.parameters import _param
from SNN2.src.params.paramHandler import ParamHandler as PH

class TestEnvConf():


    @pytest.mark.parametrize("Value, result", [
            (1, 0.16513085),
            (12, 0.47720265),
            (128, 0.18266475)])
    def test_set_rngs(self, Value, result, env_rng_param, env_tf_rng_param, dummy_log) -> None:
        env_rng_param["value"] = f"{Value}"
        env_tf_rng_param["value"] = f"{Value}"
        params = PH([env_rng_param, env_tf_rng_param], None, logger=dummy_log)
        set_rngs(params)
        np.testing.assert_almost_equal(tf.random.uniform([1]).numpy(), np.array([result], dtype=np.float32))

