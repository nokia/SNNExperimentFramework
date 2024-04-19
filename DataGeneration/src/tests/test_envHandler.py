# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import os

from Dkr5G.src.core.environment import EnvironmentHandler as ENV
from Dkr5G.src.io.configuration import ConfHandler
from Dkr5G.src.util.strings import strings as s

class TestEnvironment:

    def test_from_cfg(self, environment_configuration, env_io, env_logger):
        assert isinstance(environment_configuration, ConfHandler)
        env = ENV.from_cfg(environment_configuration, env_io, env_logger)
        assert env["id"] == "default_environment"
        assert env["test_file"] == "tests/files/test_file.txt"
        assert env["test_file_two"] == f"{env_io['tmp_value']}.txt2"
        assert env["multi_net_name"] == "Dkr5Gnet_Dkr5Gnet"
        assert env["path_network_name"] == f"{env_io['tmp_value']}_Dkr5Gnet"
        assert env["test_file_three"] == f"{env_io['tmp_value']}.Dkr5Gnet.txt2"
        directory_path = os.getcwd()
        assert env["pkg_resource_name"] == f"{directory_path}/Dkr5G"

    def test_not_handled_obj(self, environment_configuration, env_io, env_logger):
        assert isinstance(environment_configuration, ConfHandler)
        env = ENV.from_cfg(environment_configuration, env_io, env_logger)
        with pytest.raises(Exception):
            env.define_object({s.env_type_key: "Not handled", "value": "unknown"})

    def test_get_keyError(self, environment_configuration, env_io, env_logger):
        assert isinstance(environment_configuration, ConfHandler)
        env = ENV.from_cfg(environment_configuration, env_io, env_logger)
        with pytest.raises(KeyError):
            env["NotExisting item"]

    @pytest.mark.parametrize(("pth", "expected"), [
            ("hello", "hello"),
            ("hello{id}", "hellodefault_environment"),
            ("{id}hello", "default_environmenthello"),
            ("{id}hello{id}", "default_environmenthellodefault_environment"),
        ])
    def test_evaluate_path(self, pth, expected, environment_configuration, env_io, env_logger):
        assert isinstance(environment_configuration, ConfHandler)
        env = ENV.from_cfg(environment_configuration, env_io, env_logger)
        assert env.evaluate_path(pth) == expected

    def test_evaluate_path_error(self, environment_configuration, env_io, env_logger):
        assert isinstance(environment_configuration, ConfHandler)
        env = ENV.from_cfg(environment_configuration, env_io, env_logger)
        with pytest.raises(KeyError):
            env.evaluate_path("key{keyNotKnow}")

    def test_env_str(self, environment_configuration, env_io, env_logger):
        assert isinstance(environment_configuration, ConfHandler)
        env = ENV.from_cfg(environment_configuration, env_io, env_logger)
        directory_path = os.getcwd()
        result = f"""id: default_environment
network: 192.168.11.0/24
network_name: Dkr5Gnet
test_file: File handler: tests/files/test_file.txt
test_file_two: File handler: tests/files/test_file.txt.txt2
multi_net_name: Dkr5Gnet_Dkr5Gnet
path_network_name: tests/files/test_file.txt_Dkr5Gnet
test_file_three: File handler: tests/files/test_file.txt.Dkr5Gnet.txt2
pkg_resource_name: Directory Handler: {directory_path}/Dkr5G
tst_addr: 192.168.11.10\n"""
        assert str(env) == result
