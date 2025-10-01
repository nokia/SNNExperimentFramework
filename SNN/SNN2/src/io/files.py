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
files module
============

Use this module to handle files

"""

from __future__ import annotations

import os

from SNN2.src.io.directories import DirectoryHandler as DH

from typing import List, Optional

class FileHandler:
    """FileHandler.
    Class to manage single files
    """


    def __init__(self, filepath: str,
                 create: bool = True):
        """__init__.
        If the file passed doesn't exists it will be
        created, respecting the flag create.
        If the flag is false and the file doesn't
        exists an exception will be raised

        Parameters
        ----------
        filepath : str
            path to the file to use
        create : bool
            If it is possible it will create the file, if it doesn't exists

        Raises
        -----
        FileNotFoundError if create False and the file doesn't exists
        """

        self.__filename: str = filepath.split("/")[-1]
        self.__path: str = os.path.join("/".join(filepath.split("/")[:-1]))
        self.__filepath: str = filepath
        # Check if the path exists
        if not os.path.exists(self.__filepath):
            if create:
                os.mknod(self.__filepath)
            else:
                raise FileNotFoundError("{}".format(self.__filepath))

    def __str__(self) -> str:
        """__str__.
        print information about the file handler

        Parameters
        ----------

        Returns
        -------
        str

        """
        return f"File handler: {self.path}"

    @property
    def path(self) -> str:
        """path.

        Parameters
        ----------

        Returns
        -------
        str
            the complete file path to the destination file

        """
        return self.__filepath

    @property
    def folderPath(self) -> str:
        """folderPath.
        Get the folder path of a file

        Parameters
        ----------

        Returns
        -------
        str the path to the file, without the file, only the folders

        """
        return self.__path

    def get(self, extension: str) -> str:
        """get.
        Permits to get a file path with the same file name as the handled one
        plus an appendix, remember the type of the file, the part after the '.'
        will be removed

        Parameters
        ----------
        extension : str
            string to use instead of the default desinence of the file

        Returns
        -------
        str the filepath modified

        """
        pth = self.__filepath.split(".")[0]
        pth += f".{extension}"
        return pth

    @staticmethod
    def exists(filepath: str) -> bool:
        """exists.
        Return if a file exists or not

        Parameters
        ----------
        filepath : str
            filepath

        Returns
        -------
        bool

        """
        return os.path.exists(filepath)

    @staticmethod
    def detect(dir: DH) -> List[FileHandler]:
        return [FileHandler(file) for file in dir.files]

    @staticmethod
    def hash_path(path: str, hash: Optional[str] = None) -> str:
        if hash is None:
            return path
        lpath = path.split('.')
        return f"{lpath[0]}-{hash}.{lpath[1]}"

    @staticmethod
    def extend_path(path: str, extend: str) -> str:
        lpath = path.split('.')
        return f"{lpath[0]}-{extend}.{lpath[1]}"


    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return self

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return not self == other

