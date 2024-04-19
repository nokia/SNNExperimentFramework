# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Experiment module
=================

Module used to manage the experiments

"""

import sys
from tqdm import tqdm

from Dkr5G.src.core.scheduler import scheduler
from Dkr5G.src.core.event import Event, PostEvent
from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.core.graphHandler import GraphHandler as GH
from Dkr5G.src.core.dockerWrapper import DockerWrapper as DkrW
from Dkr5G.src.core.environment import EnvironmentHandler as ENV

from typing import List, Dict, Any, Callable

class ExperimentHandler():

    def __init__(self, graph: GH,
                 env: ENV,
                 io: IOH,
                 logger: LH):
        self.logger = logger
        self.lg_head = self.__class__.__name__
        self.__graph = graph
        self.__env = env
        self.__io = io
        self.__dockerWrapper = DkrW(self.logger)
        self.__scheduler = scheduler([])
        self.__postScheduler = scheduler([])
        self.logger(self.lg_head, f"Experiment handler loaded", level=LH.DEBUG)
        self.docker = None

    def evaluate(self, object: str) -> str:
        return self.__graph.evaluateObj(object, self.env, self.io)

    def scheduleJobs(self, events: List[Dict[str, Any]]):
        self.logger(self.lg_head, f"Job scheduling started, number of jobs to schedule: {len(events)}")
        for event in events:
            event[s.eve_command] = self.evaluate(event[s.eve_command])
            ev = Event(event)
            self.__scheduler.append(ev)
        self.__scheduler.sort()
        self.__scheduler.adapt_times()

    def schedulePostJobs(self, events: List[Dict[str, Any]]):
        self.logger(self.lg_head, "Post jobs scheduling started")
        for event in events:
            event[s.eve_command] = self.evaluate(event[s.eve_command])
            ev = PostEvent(event)
            self.__postScheduler.append(ev)
        self.__postScheduler.sort()
        self.__postScheduler.adapt_times()

    def executeAll(self, debug: bool = False) -> None:
        if debug:
            self.logger(self.lg_head, "Execution in debug mode")
        self.logger(self.lg_head, "Execution of all the jobs scheduled")
        t = tqdm(total=sum([x.start_time for x in self.__scheduler]))
        for job in self.__scheduler:
            self.logger(self.lg_head, f"next Job: {job}", level=LH.DEBUG)
            if debug:
                print(job.command)
            else:
                job.run()
            t.update(job.start_time)
            self.logger(self.lg_head, f"Job executed: {job}", level=LH.DEBUG)
        t.close()
        self.logger(self.lg_head, "Execution of all the jobs terminated")

    def postExecution(self, debug: bool = False) -> None:
        if debug:
            self.logger(self.lg_head, "Execution in debug mode")
        self.logger(self.lg_head, "Execution of all the post jobs scheduled")
        t = tqdm(total=sum([x.start_time for x in self.postScheduler]))
        for job in self.postScheduler:
            self.logger(self.lg_head, f"next Job: {job}", level=LH.DEBUG)
            if debug:
                print(job.command)
            else:
                output = job.run()
            t.update(job.start_time)
            self.logger(self.lg_head, f"Job executed: {job}", level=LH.DEBUG)
        t.close()
        self.logger(self.lg_head, "Execution of all the post jobs terminated")

    def start_env(self):
        self.logger(self.lg_head, f"Starting the environment {self.env[s.env_id]}")
        docker_compose_file = self.io[s.docker_file]
        self.logger(self.lg_head, f"Docker compose file to load: {docker_compose_file}", level=LH.DEBUG)
        self.__dockerWrapper.composer_up(docker_compose_file)

    def stop_env(self, debug: bool = False):
        if debug:
            while True:
                print("Please use ctrl+D to exit the debug mode and terminate the environment")
                line = sys.stdin.readline()
                if not line:
                    print("---- Termination in progress ----")
                    break
        self.logger(self.lg_head, f"Stopping the environment {self.env[s.env_id]}")
        docker_compose_file = self.io[s.docker_file]
        self.logger(self.lg_head, f"Docker compose file to stop: {docker_compose_file}", level=LH.DEBUG)
        self.__dockerWrapper.composer_down(docker_compose_file)

    @property
    def io(self):
        return self.__io

    @property
    def env(self):
        return self.__env

    @property
    def scheduler(self):
        return self.__scheduler

    @property
    def postScheduler(self):
        return self.__postScheduler
