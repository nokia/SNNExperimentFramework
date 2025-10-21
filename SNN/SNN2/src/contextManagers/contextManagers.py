# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Context Managers Module
=======================

This module provides context managers and utilities for timing code execution
and handling augmented class names within the SNN2 framework.

The module contains both function-based and class-based context managers that
can be used to measure execution times of code blocks, with optional logging
support. These tools are essential for performance monitoring and profiling
in neural network training and data processing pipelines.

Functions
---------
get_realName : function
    Extract the real name from objects, handling augmented classes.
timeit_cnt : contextmanager
    Context manager for timing code execution with logging support.

Classes
-------
TimeitContext : class
    Reusable context manager for collecting timing statistics across multiple runs.

Examples
--------
Basic timing with logging:

>>> with timeit_cnt("data_processing"):
...     # Some data processing code
...     process_data()
data_processing execution time: 1234.567 ms

Reusable timing context:

>>> timer = TimeitContext()
>>> for i in range(3):
...     with timer():
...         time.sleep(0.1)
>>> print(f"Mean execution time: {timer.mean:.3f}s")
Mean execution time: 0.100s

Notes
-----
This module integrates with the SNN2 logging system through the f_logger
decorator, providing consistent logging behavior across the framework.
All timing measurements are performed using Python's time.time() function
for high-resolution timing.

See Also
--------
SNN2.src.decorators.decorators : Decorator functions including f_logger
SNN2.src.io.logger : Logging utilities and LogHandler class
"""

from time import time
from contextlib import contextmanager
from typing import Any
from SNN2.src.decorators.decorators import f_logger

from SNN2.src.io.logger import LogHandler

def get_realName(obj: Any):
    """
    Get the real name of an object, handling augmented classes.

    Parameters
    ----------
    obj : Any
        The object whose real name should be retrieved.

    Returns
    -------
    str
        The real name of the object. If the object is an augmented class
        with a custom name, returns the passed name, otherwise returns
        the qualified name.

    Notes
    -----
    This function is designed to handle special cases where objects
    have been augmented and their original name has been stored in
    a `c_passed_name` attribute.
    """
    if obj.__name__ == "augmented_cls":
        return obj.c_passed_name
    return obj.__qualname__


@contextmanager
@f_logger
def timeit_cnt(name,  active: bool = True, **kwargs):
    """
    Context manager for timing code execution with logging support.

    This context manager measures the execution time of a code block
    and optionally logs the result using the provided logger.

    Parameters
    ----------
    name : str
        Name identifier for the timed operation, used in log messages.
    *args : tuple
        Variable length argument list (not used in current implementation).
    active : bool, default=True
        Whether to actively log and print timing results.
    **kwargs : dict
        Keyword arguments containing logger and write_msg from f_logger decorator.

    Yields
    ------
    None
        Control is yielded to the code block being timed.

    Notes
    -----
    This function is decorated with @f_logger which automatically provides
    logger and write_msg in kwargs. The timing results are both logged
    and printed to stdout when active=True.

    Examples
    --------
    >>> with timeit_cnt("my_operation"):
    ...     # Some time-consuming operation
    ...     time.sleep(1)
    my_operation execution time: 1000.123 ms
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]

    start_time = time()
    yield
    delta_t = time() - start_time
    if active:
        write_msg(f"{name} execution time: {float(delta_t*1000):.3f} ms", LogHandler.DEBUG)
        print(f"{name} execution time: {float(delta_t*1000):.3f} ms")

class TimeitContext:
    """
    A reusable context manager for timing multiple code executions.

    This class provides a context manager that can be reused multiple times
    to measure execution times and collect statistics about performance.

    Attributes
    ----------
    all_times : list of float
        List storing all recorded execution times in seconds.
    execution_time : float or None
        The most recent execution time in seconds.
    n_runs : int
        Number of times the context manager has been used.

    Examples
    --------
    >>> timer = TimeitContext()
    >>> with timer():
    ...     time.sleep(0.1)
    >>> print(f"Execution time: {timer.execution_time:.3f}s")
    >>> print(f"Mean time: {timer.mean:.3f}s")
    """

    def __init__(self):
        """
        Initialize the TimeitContext instance.

        Sets up empty containers for timing data and counters.
        """
        self.all_times = []
        self.execution_time = None
        self.n_runs = 0

    @contextmanager
    def __call__(self):
        """
        Context manager method for timing code execution.

        This method makes the TimeitContext instance callable as a context
        manager. It records the start and end times, calculates execution
        time, and updates internal statistics.

        Yields
        ------
        TimeitContext
            Returns self to allow access to timing data within the context.

        Notes
        -----
        The execution time is automatically calculated and stored in both
        execution_time (most recent) and all_times (complete history).
        The n_runs counter is incremented after each execution.
        """
        start_time = time()  # Record the start time
        try:
            yield self  # Yield control to the block of code inside the context
        finally:
            end_time = time()  # Record the end time
            self.execution_time = end_time - start_time  # Calculate execution time
            self.all_times.append(self.execution_time)
            self.n_runs += 1

    @property
    def mean(self):
        """
        Calculate the mean execution time across all runs.

        Returns
        -------
        float
            The average execution time in seconds across all recorded runs.

        Raises
        ------
        ZeroDivisionError
            If no runs have been recorded (all_times is empty).

        Notes
        -----
        This property computes the arithmetic mean of all execution times
        stored in the all_times list.
        """
        return sum(self.all_times) / len(self.all_times)
