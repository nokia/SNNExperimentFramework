# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
IO Module
=========

Use this module to control io objects from the configuration file

"""

from __future__ import annotations

import re
import pkg_resources

from typing import List, Dict, TypeVar, Union, Tuple
from SNN2.src.util.strings import s
from SNN2.src.io.files import FileHandler as fh
from SNN2.src.io.directories import DirectoryHandler as dh
from SNN2.src.io.configuration import ConfHandler as CH
from SNN2.src.io.commandInterpolation import fn as CI


T = TypeVar('T')

class IOHandler:
    """IOHandler.
    Class used to handle and control io objects like file and folders
    """


    def __init__(self, object: List[dict]):
        """__init__.

        Parameters
        ----------
        object : List[dict]
            Dictionary list of objects that must be handled
        """
        self.objects = {}
        for obj in object:
            self.objects[obj[s.io_name_key]] = self.define_object(obj)

    @classmethod
    def from_cfg(cls, conf: CH) -> IOHandler:
        items = []
        for key in conf.cfg.keys():
            if len(conf.cfg.items(key)) > 0:
                d = dict(conf.cfg.items(key))
                if d[s.io_type_key] in s.folder_obj_types or\
                   d[s.io_type_key] in s.file_obj_types:
                    d[s.io_name_key] = key
                    if s.io_exists_key in d:
                        d[s.io_exists_key] = conf.cfg.getboolean(key, s.io_exists_key)
                    else:
                        raise KeyError(f"Configuration io objects must contain the {s.io_exists_key} key")

                    if d[s.io_path_key] == "__name__":
                        d[s.io_path_key] = pkg_resources.resource_filename("SNN2.main", "")
                    items.append(d)
        return cls(items)

    def define_object(self, obj: dict) -> dh | fh:
        """define_object.
        Function used to define a single object and associate it to the
        DirectoryHandler or the FileHandler

        Parameters
        ----------
        obj : dict
            Single object that must be handled
        """
        path = self.evaluate_path(obj[s.io_path_key])
        create = not obj[s.io_exists_key]
        if obj[s.io_type_key] in s.folder_obj_types:
            return dh(path, create=create)
        elif obj[s.io_type_key] in s.file_obj_types:
            return fh(path, create=create)
        else:
            raise Exception(f"object {obj[s.io_name_key]} type {obj[s.io_type_key]} not handled")

    def evaluate_obj(self, obj: str) -> Tuple[str, str]:
        obj = obj.replace('{', '').replace('}', '')
        if obj in self.objects.keys():
            return obj, self.get(obj)
        return obj[1::], CI(obj)

    def evaluate_objs(self, objs: List[str]) -> List[Tuple[str, str]]:
        return [self.evaluate_obj(obj) for obj in objs]

    def evaluate_path(self, pth: str) -> str:
        """evaluate_path.
        Evaluate a path if it contains other objects in it

        Parameters
        ----------
        pth : str
            pth

        Returns
        -------
        str

        """
        result = re.findall(r"\{[^\}]*\}", pth)
        res = {x[0]: x[1] for x in self.evaluate_objs(result)}
        pth = pth.replace('!', '').format(**res)
        return pth

    def get(self, elem: T) -> str:
        """get.
        Returns the path of an object

        Parameters
        ----------
        elem : str
            elem

        Returns
        -------
        str

        """
        return self.objects[elem].path

    def get_handler(self, elem: str) -> Union[fh, dh]:
        """get.
        Returns the handler of an object

        Parameters
        ----------
        elem : str
            elem

        Returns
        -------
        str

        """
        return self.objects[elem]


    def __getitem__(self, item: str) -> str:
        """__getitem__.

        Parameters
        ----------
        item : str
            item that should be returned

        Returns
        -------
        str

        """
        return self.get(item)

    def check(self, key: str) -> bool:
        return key in self.objects.keys()

    def __str__(self) -> str:
        """__str__.
        Return an object in string format

        Returns
        -------
        str

        """

        res = ""
        for key in self.objects:
            res += "{}: {}\n".format(key, self.objects[key])
        return res
