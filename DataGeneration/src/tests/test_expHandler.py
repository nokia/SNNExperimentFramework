# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from Dkr5G.src.core.experiment import ExperimentHandler as EXP
from Dkr5G.src.util.strings import strings as s

class TestExpHandler:

    def test_scheduleJobs(self,
                          exp_graph,
                          exp_env,
                          exp_io,
                          exp_logger,
                          exp_events):
        exp = EXP(exp_graph, exp_env, exp_io, exp_logger)
        assert isinstance(exp, EXP)
        exp.scheduleJobs(exp_events[s.events_key])
        assert "tests/files/test_log.log" in exp.scheduler[0].command
        assert "Dkr5Gnet" in exp.scheduler[1].command
        assert "192.168.11.10" in exp.scheduler[2].command

    @pytest.mark.parametrize(("object", "answer"), [
            ("hello", "hello"),
            ("{Log}", "tests/files/test_log.log"),
            ("{network_name}", "Dkr5Gnet"),
            ("{test[ipv4]}", "192.168.11.10"),
            ("text-before {Log}", "text-before tests/files/test_log.log"),
            ("text-before {network_name}", "text-before Dkr5Gnet"),
            ("text-before {test[ipv4]}", "text-before 192.168.11.10"),
            ("text-before {Log} text-after", "text-before tests/files/test_log.log text-after"),
            ("text-before {network_name} text-after", "text-before Dkr5Gnet text-after"),
            ("text-before {test[ipv4]} text-after", "text-before 192.168.11.10 text-after")
        ])
    def test_evaluate(self,
                      exp_graph,
                      exp_env,
                      exp_io,
                      exp_logger,
                      object,
                      answer):
        exp = EXP(exp_graph, exp_env, exp_io, exp_logger)
        assert isinstance(exp, EXP)
        assert exp.evaluate(object) == answer

    def test_scheduleJobsPostExec(self,
                                  exp_graph,
                                  exp_env,
                                  exp_io,
                                  exp_logger,
                                  exp_PostEvents):
        exp = EXP(exp_graph, exp_env, exp_io, exp_logger)
        assert isinstance(exp, EXP)
        exp.schedulePostJobs(exp_PostEvents[s.events_postDocker_key])
        assert not "docker" in exp.postScheduler[0].command
        assert "echo \"Hello World!\"" in exp.postScheduler[0].command

