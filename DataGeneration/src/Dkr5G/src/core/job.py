# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Job module
==========

Module used to mange Jobs derived by an event

"""

from __future__ import annotations

import subprocess
import time

from subprocess import CompletedProcess

from Dkr5G.src.core.event import Event
from typing import Union

class Job(object):

    def __init__(self, eve: Event):
        self.__job_id = eve.id
        self.reference = eve.reference
        self.start_time = eve.start_time
        self.command = eve.command
        self.kill_time = eve.kill_time

    def run(self) -> Union[CompletedProcess, None]:
        time.sleep(int(self.start_time))
        if self.reference == "None":
            return None
        proc = subprocess.run(self.command, shell=True)
        return proc

    def __eq__(self, other: Job) -> bool:
        return self.start_time == other.start_time and self.kill_time == other.kill_time

    def __ne__(self, other: Job) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Job) -> bool:
        if self.start_time < other.start_time:
            return True
        elif self.start_time == other.start_time and \
                self.kill_time < other.kill_time:
            return True
        return False

    def __le__(self, other: Job) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: Job) -> bool:
        return not self.__lt__(other) and self.__ne__(other)

    def __ge__(self, other: Job) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    @property
    def id(self) -> int:
        return self.__job_id

    def __str__(self) -> str:
        return f"{self.id}: {self.start_time} -> {self.kill_time}: {self.command}"

