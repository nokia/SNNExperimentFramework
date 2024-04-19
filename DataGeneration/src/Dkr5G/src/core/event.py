# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Event module
============

Module used to mange events required by the user

"""

from Dkr5G.src.core.dockerWrapper import DockerWrapper as DW
from Dkr5G.src.util.strings import strings as s

from typing import Dict, Any

class Event():
    """Event.
    Class used to manage the events that are executed during the experimentation
    """


    ID: int = 0

    @classmethod
    def increment(cls) -> int:
        """increment.
        Method used to assign a unique incremental id to each event

        Returns
        -------
        int
            The next integer id that should be used
        """
        cls.ID += 1
        return cls.ID

    def __init__(self, obj: Dict[str, Any]):
        """__init__.
        Initializer for the event class, it receives as input
        a dictionary that must contain all the objects of an event
        the node, the command, the start time and the kill time.
        The kill timer starts after the command has been started

        Parameters
        ----------
        obj : Dict[str, Any]
            obj that describes the event
        """
        self.id = Event.increment()
        self.reference = obj[s.eve_node]
        self.script = obj[s.eve_command]
        self.start_time = obj[s.eve_st]
        self.kill_time = obj[s.eve_kt]
        self.kill_limit = 1

        if not self.check_command():
            raise Exception(f"Potential HARMFUL COMMAND detected {self.script} in {self.reference}")

    def check_command(self) -> bool:
        """check_command.
        Used to prevent harmful commands

        Parameters
        ----------

        Returns
        -------
        bool

        """
        harmful_commands = [
                    "rm -rf /;"
                ]
        if  any(x in self.script for x in harmful_commands):
            return False
        return True

    @property
    def command(self) -> str:
        """command.
        Property to obtain the command associated with one event
        If the reference is 'host' then the command is returned
        as it is, otherwise it's encapsulated
        in the docker formatter for a docker container

        Parameters
        ----------

        Returns
        -------
        str the command required

        """
        if self.reference == "host":
            return self.script
        return DW.docker_command_format(self.script, self.reference, self.kill_limit, self.kill_time)

    def __str__(self) -> str:
        """__str__.

        Parameters
        ----------

        Returns
        -------
        str

        """
        return f"{self.id}: {self.reference} <- {self.command} at {self.start_time}"

class PostEvent(Event):
    """PostEvent.
    This class specifically manages events that should be executed after the
    docker environment has been closed
    """


    def __init__(self, obj: Dict[str, Any]):
        """__init__.

        Parameters
        ----------
        obj : Dict[str, Any]
            obj
        """
        obj[s.eve_node] = "Post"
        super().__init__(obj)

    @property
    def command(self) -> str:
        """command.
        Returns the command with other modifications

        Returns
        -------
        str
        """
        return self.script
