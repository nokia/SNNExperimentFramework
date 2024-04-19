# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
files module
============

Use this module to handle files

"""

from __future__ import annotations

import os

from Dkr5G.src.io.directories import DirectoryHandler as DH
from Dkr5G.src.util.strings import strings as s

from typing import Tuple

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
    def detect(dirPath: DH) -> Tuple[FileHandler, FileHandler, FileHandler]:
        return (FileHandler(os.path.join(dirPath.path, s.conf_default_folder, s.conf_default_file_name)),
                FileHandler(os.path.join(dirPath.path, s.graph_default_folder, s.graph_default_file_name)),
                FileHandler(os.path.join(dirPath.path, s.events_default_folder, s.events_default_file_name)))

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return self

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return not self == other
