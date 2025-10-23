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
Directory module
================

Use this module to handle directories

"""

import os
import shutil

from typing import List

class DirectoryHandler:
    """DirectoryHandler.
    Class used to manage directories
    """


    def __init__(self, path: str, create: bool = True):
        """__init__.
        if the directory passed doesn't exists and create is
        false then an exception is generated

        Parameters
        ----------
        path : str
            path to the directory position
        create : bool
            create the directory if it doesn't exists

        Raises
        ------
        FileNotFoundError
        """
        self.__path = path
        if not os.path.exists(self.__path):
            if create:
                os.makedirs(self.__path)
            else:
                raise FileNotFoundError(f"{self.__path} not Found")

    @property
    def path(self) -> str:
        """path of the directory managed

        Parameters
        ----------

        Returns
        -------
        str
            the path to the directory

        """
        return self.__path

    @property
    def files(self) -> List[str]:
        """Returns the list of files in the directory controlled.

        Parameters
        ----------

        Returns
        -------
        List[str]

        """
        res = []
        for file in os.scandir(self.path):
            res.append(f"{self.path}/{file.name}")
        return res

    @classmethod
    def clear(cls, path: str) -> None:
        for elem in os.scandir(path):
            try:
                if os.path.isfile(elem) or os.path.islink(elem):
                    os.unlink(elem)
                elif os.path.isdir(elem):
                    shutil.rmtree(elem)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (elem, e))

    def __str__(self) -> str:
        """__str__.

        Parameters
        ----------

        Returns
        -------
        str

        """
        return "Directory Handler: {}".format(self.path)
