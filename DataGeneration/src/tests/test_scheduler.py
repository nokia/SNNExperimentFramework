# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from Dkr5G.src.core.scheduler import scheduler
from Dkr5G.src.core.scheduler import Job

class TestScheduler:

    def test_adapt_times(self, test_scheduler):
        assert isinstance(test_scheduler, scheduler)
        test_scheduler.adapt_times()
        for elem in test_scheduler:
            assert isinstance(elem, Job)
        assert test_scheduler[0].start_time == 10
        assert test_scheduler[1].start_time == 5
        assert test_scheduler[2].start_time == 2
        assert test_scheduler[3].start_time == 3

    def test_init(self, test_events):
        tmp_sch = scheduler(test_events)
        sch = scheduler(tmp_sch)
        for elem in sch:
            assert isinstance(elem, Job)
        assert sch[0].start_time == 10
        assert sch[1].start_time == 15
        assert sch[2].start_time == 17

    def test_adapt_times_empty(self, empty_scheduler):
        assert empty_scheduler == empty_scheduler.adapt_times()

    def test_add(self, empty_scheduler, test_events):
        sch = empty_scheduler + test_events
        for elem in sch:
            assert isinstance(elem, Job)
        assert len(sch) == 3
        assert sch[0].start_time == 10
        assert sch[1].start_time == 15
        assert sch[2].start_time == 17

    def test_add_append(self, empty_scheduler, test_events):
        sch = empty_scheduler
        for elem in test_events:
            sch += elem
        for elem in sch:
            assert isinstance(elem, Job)
        assert len(sch) == 3
        assert sch[0].start_time == 10
        assert sch[1].start_time == 15
        assert sch[2].start_time == 17


    def test_iadd(self, empty_scheduler, test_events):
        empty_scheduler +=test_events
        for elem in empty_scheduler:
            assert isinstance(elem, Job)
        assert len(empty_scheduler) == 3
        assert empty_scheduler[0].start_time == 10
        assert empty_scheduler[1].start_time == 15
        assert empty_scheduler[2].start_time == 17

    def test_append_error(self, empty_scheduler):
        with pytest.raises(ValueError):
            empty_scheduler.append(12)

    def test_str(self, test_scheduler):
        s = """Scheduler list: \n['27: 10 -> 10: docker exec -d test sh -c "timeout -k 1 10 bash -c  \\\'test\\\'"', '28: 15 -> 2: docker exec -d test sh -c "timeout -k 1 2 bash -c  \\\'test\\\'"', '29: 17 -> 1: docker exec -d test sh -c "timeout -k 1 1 bash -c  \\\'test\\\'"']"""
        assert str(test_scheduler) == s
