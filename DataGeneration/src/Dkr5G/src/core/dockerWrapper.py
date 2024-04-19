# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
DockerWrapper module
====================

Module used to manage the docker commands that needs to be executed
to load the environment and the execute the scheduled events.

"""

import shlex
import subprocess

from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.logger import LogHandler as LH
from typing import List

class DockerWrapper():

    def __init__(self, logger: LH):
        self.logger = logger
        self.stdout = subprocess.DEVNULL
        self.stderr = subprocess.DEVNULL
        self.logger(self.__class__.__name__, "Docker wrapper started", level=LH.DEBUG)

    def execute(self, command: List[str]) -> int:
        process = subprocess.Popen(command, stdout=self.stdout, stderr=self.stderr)
        exit_code = process.wait()
        return exit_code

    def composer_up(self, composer_file: str) -> None:
        command = shlex.split(s.docker_up.format(file=composer_file))
        self.logger(self.__class__.__name__, f"Starting the compose file {composer_file}")
        self.logger(self.__class__.__name__, f"command executed {command}")
        out = self.execute(command)
        self.logger(self.__class__.__name__, f"Environment started, exit code: {out}")

    def composer_down(self, composer_file) -> None:
        command = shlex.split(s.docker_down.format(file=composer_file))
        self.logger(self.__class__.__name__, f"Stopping the compose file {composer_file}")
        out = self.execute(command)
        self.logger(self.__class__.__name__, f"Environment stopped, exit code: {out}")

    @classmethod
    def docker_command_format(cls, command: str,
                              container: str,
                              kill_limit: int,
                              kill_time: int) -> str:
        return s.docker_command.format(container,
                                       s.timeout_command.format(kill_limit, kill_time, command))
