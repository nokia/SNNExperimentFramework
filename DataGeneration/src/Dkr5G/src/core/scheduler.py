# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Scheduler module
================

Module used to mange the scheduler used during the experiments

"""

from __future__ import annotations

from Dkr5G.src.util.strings import strings as s
from collections import UserList
from Dkr5G.src.core.event import Event
from Dkr5G.src.core.job import Job

from typing import Iterable, Union

class scheduler(UserList):
    """scheduler.
    This class is responsible for scheduler objects.
    Schedulers are lists of events that will be sorted
    using the event starting time and kill time.
    Such that events that have a lower starting time will be executed ahead.
    The starting time will be adapted from the user absolute one
    to a relative one in respect of the previous event.
    Such that is possible to use the starting time as a delta
    to wait between jobs before the execution.
    """


    def __init__(self, info=()):
        """__init__.

        Parameters
        ----------
        info :
            list of events or a scheduler itself, to generate a new
            scheduler that contains all the data from the first one
        """
        super().__init__(self)
        if isinstance(info, scheduler):
            self.data[:] = info.data
        else:
            self.extend(info)

    def add_iterable(self, iterable: Iterable[Event]) -> scheduler:
        """add_iterable.
        Use the append method for each element inside the iterable object
        elements must be events

        Parameters
        ----------
        iterable : Iterable[Event]
            iterable to cycle in order to include all the elements.

        Returns
        -------
        scheduler
        """
        # TODO shouldn't we use extend instead of a cycle?
        for i in iterable:
            self.append(i)
        return self

    def adapt_times(self) -> None:
        """adapt_times.
        This function adapts the timers of the current
        scheduler from an absolute point of view to a relative one
        in reference of the previous job in the queue.
        It acts inplace.

        Parameters
        ----------

        Returns
        -------
        None

        """
        self.sort()
        if len(self.data) == 0:
            return
        current_time = self.data[0].start_time
        current_end_time = current_time + self.data[0].kill_time

        if len(self.data) == 1:
            return

        for event in self.data[1:]:
            event.start_time = event.start_time - current_time
            current_time += event.start_time
            possible_end_time = current_time + event.kill_time
            current_end_time = possible_end_time if possible_end_time > current_end_time else current_end_time

        self.append(Event({s.eve_node: "None",
                           s.eve_command: "None",
                           s.eve_st: current_end_time - current_time,
                           s.eve_kt: 0}))

    def __add__(self, something_new: Union[Iterable, Event]) -> scheduler:
        """__add__.
        Append something to the current scheduler
        it can be an Iterable or a single event

        Parameters
        ----------
        something_new : Union[Iterable, Event]
            what needs to be appended, it will use the extend or append method
            accordingly

        Returns
        -------
        scheduler

        """
        ret = self.copy()
        if hasattr(something_new, '__iter__'):
            return ret.extend(something_new)
        else:
            return ret.append(something_new)

    def __iadd__(self, new: Union[Iterable, Event]) -> scheduler:
        """__iadd__.
        Inplace version of __add__

        Parameters
        ----------
        new : Union[Iterable, Event]
            new

        Returns
        -------
        scheduler

        """
        return self.__add__(new)

    def append(self, something_new: Event) -> scheduler:
        """append.
        Append a single event to the current scheduler

        Parameters
        ----------
        something_new : Event
            The new event to append

        Returns
        -------
        scheduler

        """
        if not isinstance(something_new, Event):
            raise ValueError(f"{something_new} is not an event")
        self.data.append(Job(something_new))
        return self

    def extend(self, something_new: Iterable[Event]) -> scheduler:
        """extend.
        Extends the current scheduler using an iterable that contains events

        Parameters
        ----------
        something_new : Iterable[Event]
            something_new

        Returns
        -------
        scheduler

        """
        return self.add_iterable(something_new)

    def __str__(self) -> str:
        """__str__.

        Parameters
        ----------

        Returns
        -------
        str

        """
        return f"Scheduler list: \n{[str(x) for x in self.data]}"

    def __eq__(self, other) -> bool:
        return all([x == other[i] for x, i in enumerate(self.data)])
