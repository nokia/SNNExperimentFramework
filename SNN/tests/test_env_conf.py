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

