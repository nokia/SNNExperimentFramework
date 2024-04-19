# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import numpy as np

from SNN2.src.util.strings import s
from SNN2.src.params.parameters import _param
from SNN2.src.params.parameters import TypeSpecific_param
from SNN2.src.params.parameters import NumpyRng_param
from SNN2.src.params.parameters import Callable_param

class TestParameters():

    def test_init(self):
        d = {
            "name": "test",
            "type": "generic",
            "value": 12
        }
        p = _param(d)
        assert isinstance(p, _param)
        assert p.type == d["type"]
        assert p.value == d["value"]
        assert p.name == d["name"]
        assert p.logger == p.none_log

    def test_logger(self, generic_param, dummy_log):
        p = _param(generic_param, logger=dummy_log)
        assert isinstance(p, _param)

    def test_string(self, generic_param):
        p = _param(generic_param)
        assert str(p) == "Param test, type: generic, value: 12"

    def test_typeSpecific(self, generic_param):
        p = TypeSpecific_param(generic_param)
        assert isinstance(p, TypeSpecific_param)
        generic_param["type"] = "string"
        p = TypeSpecific_param(generic_param, param_type="string")
        assert isinstance(p, TypeSpecific_param)
        with pytest.raises(Exception):
            TypeSpecific_param(generic_param, param_type="NotString")

    def test_NumpyRNG(self, generic_param):
        generic_param["type"] = s.param_numpyRng_type
        p = NumpyRng_param(generic_param)
        assert isinstance(p.value, np.random._generator.Generator)
        rints = p.value.integers(low=0, high=10, size=3)
        np.testing.assert_equal(
                    np.array(rints),
                    np.array([6, 2, 9])
                )

    def test_Callable(self, generic_param, dummy_callable):
        generic_param[s.param_action_args] = (1, 2, 3)
        generic_param[s.param_action_kwargs] = {"1": 1, "2": 2, "3": 3}
        with pytest.raises(Exception):
            Callable_param(generic_param)
        p = Callable_param(generic_param, dummy_callable)
        args, kwargs = p()
        assert args == (generic_param["value"], 1, 2, 3)
        assert kwargs == {"1": 1, "2": 2, "3": 3}

    def test_arg_string(self, callable_param, dummy_callable):
        p = Callable_param(callable_param, dummy_callable)
        assert str(p) == "Param test, type: generic, value: 12 - args: (1, 2, 3) - kwargs: {'1': 1, '2': 2, '3': 3}"
