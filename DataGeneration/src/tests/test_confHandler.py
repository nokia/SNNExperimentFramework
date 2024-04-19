# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import configparser

from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.directories import DirectoryHandler as DH
from Dkr5G.src.io.configuration import ConfHandler

class TestConfHandler:

    def test_file(self, conf_file):
        conf_fh = FH(conf_file, create=False)
        conf = ConfHandler(conf_fh)
        assert isinstance(conf, ConfHandler)
        assert conf.file.path == conf_file

    def test_init_excpt(self):
        with pytest.raises(TypeError):
            ConfHandler("Hello")

    def test_read_conf(self, conf_file):
        conf_fh = FH(conf_file, create=False)
        conf = ConfHandler(conf_fh)
        assert isinstance(conf.cfg, type(configparser.ConfigParser()))
        assert conf["Log"]["file_name"] == "log.log"

    def test_update(self, conf_file, nodefault_conf_file):
        default_conf = FH(conf_file, create=False)
        not_default_conf = FH(nodefault_conf_file, create=False)
        default_conf = ConfHandler(default_conf)
        default_conf.update(not_default_conf)
        assert default_conf["Log"]["file_name"] == "log_not_default.log"
        assert default_conf["Log"]["file_location"] == "./"

    def test_update_inplace(self, conf_file, nodefault_conf_file):
        default_conf = FH(conf_file, create=False)
        not_default_conf = FH(nodefault_conf_file, create=False)
        default_conf = ConfHandler(default_conf)
        second_conf = default_conf.update(not_default_conf, inplace=False)
        assert second_conf["Log"]["file_name"] == "log_not_default.log"
        assert second_conf["Log"]["file_location"] == "./"

    def test_update_error(self, conf_file):
        default_conf = FH(conf_file, create=False)
        default_conf = ConfHandler(default_conf)
        with pytest.raises(TypeError):
            default_conf.update("string")

    def test_directory_files(self, conf_dir):
        dir = DH(conf_dir)
        conf = ConfHandler(dir)
        assert isinstance(conf, ConfHandler)
        assert conf["id"]["value"] == "default_environment"
        assert conf["test_percentage"]["value"] == "%%"
        assert conf["test_file"]["value"] == "tests/files/default_environment.txt"
        assert conf["test_file_two"]["value"] == "Dkr5Gnet.txt2"
        assert conf["multi_net_name"]["value"] == "Dkr5Gnet_Dkr5Gnet"
        assert conf["path_network_name"]["value"] == "default_environment_Dkr5Gnet_file3"
        assert conf["test_file_three"]["value"] == "Dkr5Gnet.txt2.Dkr5Gnet.txt2"
        assert conf["tst_addr"]["value"] == "192.168.11.10"

    def test_directory_files_update(self, conf_dir, other_conf_dir):
        dir = DH(conf_dir)
        secondDir = DH(other_conf_dir)
        conf = ConfHandler(dir)
        assert isinstance(conf, ConfHandler)
        conf.update(secondDir)
        assert conf["multi_net_name"]["value"] == "Dkr5Gnet_Dkr5Gnet_updated"
