# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import shlex

from Dkr5G.src.core.dockerWrapper import DockerWrapper as DW
from Dkr5G.src.util.strings import strings as s

class TestDockerWrapper:

    @pytest.mark.parametrize("command", ["hello"])
    @pytest.mark.parametrize("container", ["test_container"])
    @pytest.mark.parametrize("kill_limit", [1, 10, -2])
    @pytest.mark.parametrize("kill_time", [12, 128, -2])
    def test_docker_command_format(self, command, container, kill_limit, kill_time):
        command = "hello"
        container = "test_container"
        kill_limit = 10
        kill_time = 10
        assert DW.docker_command_format(
                    command,
                    container,
                    kill_limit,
                    kill_time
               ) == s.docker_command.format(
                       container,
                       s.timeout_command.format(
                           kill_limit,
                           kill_time,
                           command
                           )
                       )

    # @pytest.mark.parametrize("command", [
    #         "clear",
    #         "echo \"hello world\""
    #     ])
    # def test_docker_execute(self, command, test_logger):
    #     dw = DW(test_logger)
    #     command = shlex.split(command)
    #     assert dw.execute(command) == 0
