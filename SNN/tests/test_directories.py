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
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

import os
import pytest

from SNN2.src.io.directories import DirectoryHandler as DH

class TestDirectories:

    def test_directory_files(self, conf_dir):
        dir = DH(conf_dir)
        assert isinstance(dir.files, list)
        assert len(dir.files) == 9
        print(dir.files)
        assert dir.files[0] == "tests/files/conf/environment.ini"
        assert dir.files[1] == "tests/files/conf/callbacks.ini"
        assert dir.files[2] == "tests/files/conf/embedding.ini"
        assert dir.files[3] == "tests/files/conf/model.ini"
        assert dir.files[4] == "tests/files/conf/experiment.ini"
        assert dir.files[5] == "tests/files/conf/actions.ini"
        assert dir.files[6] == "tests/files/conf/layers.ini"
        assert dir.files[7] == "tests/files/conf/io.ini"
        assert dir.files[8] == "tests/files/conf/pp.ini"

    def test_create_dir(self, tmp_path):
        d = tmp_path / "test-folder"
        DH(d)
        assert os.path.exists(f"{tmp_path}/test-folder")

    def test_directory_notFound(self):
        with pytest.raises(FileNotFoundError):
            DH("DirectoryNotExisting", create=False)

    def test_str(self, tmp_path):
        d = tmp_path / "test-folder"
        d = DH(d)
        assert str(d) == f"Directory Handler: {tmp_path}/test-folder"

