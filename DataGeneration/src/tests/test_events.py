# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from Dkr5G.src.core.event import Event, PostEvent
from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.core.dockerWrapper import DockerWrapper as DW

class TestEvent:

    @pytest.mark.parametrize(("command", "st", "kt"), [
            ("hello", 12, 20)
        ])
    def test_PostEvent(self, command, st, kt):
        d = {s.eve_node: "Node",
             s.eve_command: command,
             s.eve_st: st,
             s.eve_kt: kt}
        ev = PostEvent(d)
        assert ev.command == command

    @pytest.mark.parametrize(("node", "command", "st", "kt"), [
            ("client", "hello", 12, 20),
            ("host", "hello", 12, 20)
        ])
    def test_event(self, node, command, st, kt):
        d = {s.eve_node: node,
             s.eve_command: command,
             s.eve_st: st,
             s.eve_kt: kt}
        ev = Event(d)
        if node == "host":
            assert ev.command == command
        else:
            assert ev.command == DW.docker_command_format(command, node, 1, kt)

    @pytest.mark.parametrize(("command", "st", "kt"), [
            ("rm -rf /;", 12, 20)
        ])
    def test_harmful_command(self, command, st, kt):
        d = {s.eve_node: "Node",
             s.eve_command: command,
             s.eve_st: st,
             s.eve_kt: kt}
        with pytest.raises(Exception):
            Event(d)

    @pytest.mark.parametrize(("command", "st", "kt"), [
            ("hello", 12, 20)
        ])
    def test_str(self, command, st, kt):
        d = {s.eve_node: "Node",
             s.eve_command: command,
             s.eve_st: st,
             s.eve_kt: kt}
        ev = Event(d)
        ev.id = -2
        assert str(ev) == "-2: Node <- docker exec -d Node sh -c \"timeout -k 1 20 bash -c  'hello'\" at 12"
