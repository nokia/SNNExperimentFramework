# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment Handler module
==========================

Module use to manage the environment variables

"""

from __future__ import annotations

import re
import pkg_resources

from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.files import FileHandler as fh
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.io.directories import DirectoryHandler as dh
from Dkr5G.src.io.configuration import ConfHandler

from typing import List, Dict, Any

class EnvironmentHandler():
    """EnvironmentHandler.
    Class used to handle the environment variables
    """

    def __init__(self, env: List[Dict[str, Any]],
                 io_handler: IOH,
                 logger: LH):
        self.logger = logger
        self.io = io_handler
        self.objects = {}
        for obj in env:
            self.objects[obj[s.env_name_key]] = self.define_object(obj)

        self.logger.write(self.__class__.__name__, f"Environment loaded {self.objects}")

    @classmethod
    def from_cfg(cls, conf: ConfHandler, io: IOH, logger: LH) -> EnvironmentHandler:
        items = []
        for key in conf.cfg.keys():
            if len(conf.cfg.items(key)) > 0:
                d = dict(conf.cfg.items(key))
                d[s.io_name_key] = key
                if s.io_exists_key in d:
                    d[s.io_exists_key] = conf.cfg.getboolean(key, s.io_exists_key)
                if s.env_value_key in d and d[s.env_value_key] == "__name__":
                    d[s.env_value_key] = pkg_resources.resource_filename("Dkr5G", "")
                items.append(d)
        return cls(items, io, logger)

    def define_object(self, obj: dict):
        self.logger.write(self.__class__.__name__, f"obj: {obj}", LH.DEBUG)
        create = False
        if s.io_exists_key in obj.keys():
            create = not obj[s.io_exists_key]

        if obj[s.env_type_key] in s.folder_obj_types:
            path = self.evaluate_path(obj[s.env_value_key])
            self.logger.write(self.__class__.__name__, f"obj its a path {path}", level=LH.DEBUG)
            return dh(path, create=create)

        elif obj[s.env_type_key] in s.file_obj_types:
            path = self.evaluate_path(obj[s.env_value_key])
            self.logger.write(self.__class__.__name__, f"obj its a file {path}",  level=LH.DEBUG)
            return fh(path, create=create)

        elif obj[s.env_type_key] in s.env_types:
            value = self.evaluate_path(obj[s.env_value_key])
            self.logger.write(self.__class__.__name__, f"obj its a env variable {value}",  level=LH.DEBUG)
            return value

        else:
            raise Exception(f"object {obj[s.io_name_key]} type {obj[s.io_type_key]} not handled")

    def evaluate_path(self, pth: str) -> str:
        """evaluate_path.
        Evaluate a path if it contains other objects in it
        It can actually evaluate other strings than
        paths.
        If the string given doesn't contain references
        to other components then it would be left unchanged.
        If an object inside is unknown it will raise an exception

        Parameters
        ----------
        pth : str
            pth

        Returns
        -------
        str

        Raises
        ------
        KeyError if the object is unknown

        """
        result = re.findall(r"\{[^\}]*\}", pth)
        result = [element.replace('{', '').replace('}', '') for element in result]
        result_d = {}
        for elem in result:
            if elem in self.io.objects.keys():
                result_d[elem] = self.io[elem]
            elif elem in self.objects.keys():
                result_d[elem] = self.get(elem)
            else:
                raise KeyError(f"Unexpected key {elem} not found in both local elements and io elements")
        pth = pth.format(**result_d)
        return pth

    def get(self, elem: str) -> str:
        """get.
        Returns the an object given the str

        Parameters
        ----------
        elem : str
            elem

        Returns
        -------
        str

        """
        if not elem in self.objects:
            raise KeyError(f"{elem} not found in the local env variables")
        if isinstance(self.objects[elem], (fh, dh)):
            return self.objects[elem].path
        return self.objects[elem]

    def __getitem__(self, item: str) -> str:
        """__getitem__.

        Parameters
        ----------
        item : object
            item that should be returned

        Returns
        -------
        object

        """
        return self.get(item)

    def check(self, key: str) -> bool:
        return key in self.objects.keys()

    def __str__(self) -> str:
        """__str__.
        Return all the objects in string format

        Returns
        -------
        str

        """

        res = ""
        for key in self.objects:
            res += f"{key}: {self.objects[key]}\n"
        return res
