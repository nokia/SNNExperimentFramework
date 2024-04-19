# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from Dkr5G.src.io.directories import DirectoryHandler as DH

from typing import Iterator

class TestDirectories:

    def test_directory_files(self, conf_dir):
        dir = DH(conf_dir)
        assert isinstance(dir.files, list)
        assert len(dir.files) == 3
        assert dir.files[0] == "tests/files/envFolder/test_env1.cfg"
        assert dir.files[1] == "tests/files/envFolder/test_env2.cfg"
        assert dir.files[2] == "tests/files/envFolder/test_env3.cfg"

    def test_directory_notFound(self):
        with pytest.raises(FileNotFoundError):
            DH("DirectoryNotExisting", create=False)
