# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani.ext@nokia.com>

"""
logger Module
=============

Use this module in order to handle the logging operations

"""

import logging

from SNN2.src.util.strings import s

class LogHandler:
    """LogHandler.
    Class used in order to handle the log saving process
    """


    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, file, level):
        """__init__.

        Parameters
        ----------
        file :
            file path where to save the log messages
        level :
            level to filter too specific messages
        """
        logging.basicConfig(filename=file, level=level, format=s.log_format)
        self.write(self.__class__.__name__, "Logger initialized", self.DEBUG)

    def __call__(self, *args, **kwargs) -> None:
        self.write(*args, **kwargs)

    def write(self, obj: str,
              message: str,
              level: int = logging.INFO) -> None:
        """write.
        Write to the file a specific message.
        It's mandatory to define the object of the message.
        Usually the object corresponds to the class that
        has called the method `self.__class__.__name__`.

        Parameters
        ----------
        obj : str
            obj of the message
        message : str
            actual message
        level : int
            level to register the message

        Returns
        -------
        None

        """
        message = f"{obj} - {message}"

        if level == self.DEBUG:
            logging.debug(message)
        elif level == self.INFO:
            logging.info(message)
        elif level == self.WARNING:
            logging.warning(message)
        elif level == self.ERROR:
            logging.error(message)
        elif level == self.CRITICAL:
            logging.critical(message)

    @classmethod
    def findLL(cls, level: int) -> int:
        """findLL.
        Given an integer it returns the correct class level to asociate with it

        Parameters
        ----------
        level : int
            level

        Returns
        -------
        int

        """
        if level == 1:
            return cls.CRITICAL
        if level == 2:
            return cls.ERROR
        if level == 3:
            return cls.WARNING
        if level == 4:
            return cls.INFO
        return cls.DEBUG
