# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration module
====================

Use this module to manage a configuration object.

"""

from __future__ import annotations
import configparser
import copy

from SNN2.src.io.directories import DirectoryHandler as DH
from SNN2.src.io.files import FileHandler as FH

from typing import Any, Union

class ConfHandler:
    """ConfHandler.

	General object Configuration handler
    It uses the module configparser to manage and update the
    current configuration.
	"""


    def __init__(self, conf_file: Union[FH, DH]):
        """__init__.

        Parameters
        ----------
        conf_file : FH
            conf_file where to find the configuration that
            needs to be loaded, the FH type is mandatory

        Raises
        ------
        TypeError
            If the FH type is not respected
        """
        self.cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        if isinstance(conf_file, FH):
            self.file = conf_file
            self.cfg.read(self.file.path)
        elif isinstance(conf_file, DH):
            self.file = conf_file.files
            self.cfg.read(self.file)
        else:
            raise TypeError(f"Conf file expected FileHandler or DirectoryHandler, obtained {type(conf_file)}")

    def update(self, conf: Union[FH, DH],
               inplace: bool = True) -> Union[None, ConfHandler]:
        """update.
        Is possible to update the current configuration
        with the content of the conf FH passed.
        The parameters inside the conf will have priority
        over the previous configuration.
        The FH `self.file` does not reflect the new FH.
        Is possible to obtain a new ConfHandler without
        performing this operation in place

        Parameters
        ----------
        conf : FH
            conf the new configuration file handler
        inplace : bool
            inplace, when active the change is done inplace
            otherwise a new copy is returned

        Returns
        -------
        Union[None, ConfHandler]

        """
        if inplace:
            if isinstance(conf, FH):
                self.cfg.read(conf.path)
            elif isinstance(conf, DH):
                self.cfg.read(conf.files)
            else:
                raise TypeError(f"Conf file expected FileHandler or DirectoryHandler, obtained {type(conf)}")
            return None
        new_cfgH = copy.deepcopy(self)
        new_cfgH.update(conf)
        return new_cfgH

    def __getitem__(self, item: str) -> Any:
        """__getitem__.
        Get one item from the `self.cfg` object
        without calling the cfg attribute.

        Parameters
        ----------
        item : str
            item to retrieve

        Returns
        -------
        Any

        """
        return self.cfg[item]
