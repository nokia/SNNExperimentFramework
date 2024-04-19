# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import datetime

from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.io.configuration import ConfHandler
from Dkr5G.src.util.strings import strings as s

class TestIOHandler:

    def test_from_cfg(self, test_configuration):
        assert isinstance(test_configuration, ConfHandler)
        io = IOH.from_cfg(test_configuration)
        assert io["Log"] == "tests/files/test_log.log"
        assert FH.exists(io["Log"])
        os.remove(io["Log"])
        assert not FH.exists(io["Log"])

    def test_from_cfg_pkg_location(self, test_cfg_pkg):
        assert isinstance(test_cfg_pkg, ConfHandler)
        ioh = IOH.from_cfg(test_cfg_pkg)
        directory_path = os.getcwd()
        assert ioh["main_folder"] == f"{directory_path}/Dkr5G"
        assert ioh["results"] == f"{ioh['main_folder']}/test_results"
        assert FH.exists(ioh["results"])
        os.rmdir(ioh["results"])
        assert not FH.exists(ioh["results"])

    def test_get_handler(self, test_configuration):
        assert isinstance(test_configuration, ConfHandler)
        io = IOH.from_cfg(test_configuration)
        assert io["Log"] == "tests/files/test_log.log"
        assert FH.exists(io["Log"])
        log_fh = io.get_handler("Log")
        assert isinstance(log_fh, FH)
        assert log_fh.path == "tests/files/test_log.log"
        os.remove(io["Log"])
        assert not FH.exists(io["Log"])

    def test_from_cfg_exists_error(self, test_cfg_pkg):
        assert isinstance(test_cfg_pkg, ConfHandler)
        del test_cfg_pkg.cfg["main_folder"]["exists"]
        with pytest.raises(KeyError):
            IOH.from_cfg(test_cfg_pkg)

    def test_define_obj_error(self, test_configuration_error):
        with pytest.raises(Exception):
            IOH.from_cfg(test_configuration_error)

    def test_str(self, test_configuration):
        assert isinstance(test_configuration, ConfHandler)
        io = IOH.from_cfg(test_configuration)
        assert str(io) == """Log: File handler: tests/files/test_log.log
log_file: File handler: tests/files/log.log
tmp_value: File handler: tests/files/test_file.txt\n"""

    def test_specialCommand(self, command_conf):
        assert isinstance(command_conf, ConfHandler)
        io = IOH.from_cfg(command_conf)
        assert io["results"] == f"{io['main_folder']}/test_results-{datetime.datetime.now().strftime('%d.%m.%Y-%H.%M.%S')}"
