# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
