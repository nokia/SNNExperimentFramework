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
Pickle handler module
=====================

Use this module to manage pickle objects.
It permits to load and save pickle objects.
It must be initialized with the variables required to create uniquePickleFiles.

"""

import os
import pickle as pkl

from tensorflow.python.framework.ops import kwarg_only

from SNN2.src.io.IOHandler import IOHandler as IOH
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.util.strings import s

from typing import Any

class PickleHandler:
    """PickleHandler.

    General Pickle object handler class
    """

    def __write_msg(self, msg: str, level: int = LH.INFO) -> None:
        """__write_msg.
        Write a message into the log file with a defined log level

        Parameters
        ----------
        msg : str
            msg to print
        level : int
            level default INFO

        Returns
        -------
        None

        """
        self.logger(f"{self.__class__.__name__}: {msg}", level)

    def __init__(self, IoH: IOH, ExpID: str, logger: LH, hash: str = "",
                 unix_time: str = ""):
        """__init__.

        Parameters
        ----------
        IoH : IOH
            Input output object from which the output pkl path is obtained
        ExpID : str
            Unique identifier of the experiments
        """
        self.output_folder: str = IoH[s.pkl_path]
        self.unix_time: str = unix_time
        self.experiment_id: str = ExpID+hash
        self.logger: LH = logger
        self.__write_msg("Pickle handler initialized")

    def __filepath(self, name: str, disable_unique_id: bool = False,
                   unix_time: bool = False) -> str:
        """__filepath.
        Function to generate the path to a pkl file given the name of the file
        without the extension

        Parameters
        ----------
        name : str
            name

        Returns
        -------
        str

        """
        file_name = f"{name}_{self.experiment_id}.pkl"
        if unix_time:
            file_name = f"{name}_{self.experiment_id}_{self.unix_time}.pkl"
        if disable_unique_id:
            file_name = f"{name}.pkl"
        return os.path.join(self.output_folder, file_name)

    def save(self, object: object, name: str,
             override: bool = False, **kwargs) -> None:
        """save.
        Function used to save an object to a file with the given name

        Parameters
        ----------
        object : object
            object that needs to be saved
        name : str
            name of the file to write
        override :
            override if true then the file will be overwritten otherwise
            an exception will be throw

        Returns
        -------
        None

        """
        file_path = self.__filepath(name, **kwargs)
        self.__write_msg(f"Saving a pickle object at the following path: {file_path}", LH.DEBUG)

        if not override and os.path.exists(file_path):
            self.__write_msg(f"{file_path} already exists and the override option is disabled", LH.ERROR)
            raise FileExistsError(f"{file_path} Already exists, pickle creation abortion")

        with open(file_path, 'wb') as file:
            pkl.dump(object, file)

        self.__write_msg(f"{file_path} written")

    def load(self, name: str, **kwargs) -> Any:
        """load.
        Given a name load the corresponding object and returns it

        Parameters
        ----------
        name : str
            name

        Returns
        -------
        object

        """
        file_path = self.__filepath(name, **kwargs)
        self.__write_msg(f"Loading a pickle object from the following path: {file_path}", LH.DEBUG)

        if not os.path.exists(file_path):
            self.__write_msg(f"{file_path} does not exists impossible loading", LH.ERROR)
            raise FileNotFoundError(f"{file_path} Not found, pickle loading aboarted")

        with open(file_path, 'rb') as file:
            self.__write_msg(f"{file_path} Loaded")
            return pkl.load(file)

    def check(self, name: str, **kwargs) -> bool:
        """check.
        Check if a given name corresponds to an existsing pickle file

        Parameters
        ----------
        name : str
            name

        Returns
        -------
        bool

        """
        file_path = self.__filepath(name, **kwargs)
        self.__write_msg(f"Checking if {file_path} exists", LH.DEBUG)
        return os.path.exists(file_path)

