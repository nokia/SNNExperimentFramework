# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import copy

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.io.directories import DirectoryHandler as DH

class TestFileHandler:

    def test_file(self, tmp_file):
        file_fh = FH(tmp_file, create=False)
        assert isinstance(file_fh, FH)
        assert file_fh.path == tmp_file

    def test_file_notExisting(self, tmp_files_path):
        new_file = str(tmp_files_path / "new_file.txt")
        file_fh = FH(new_file, create=True)
        assert isinstance(file_fh, FH)
        assert file_fh.path == new_file

    def test_folderPath(self, tmp_file):
        file_fh = FH(tmp_file, create=False)
        assert file_fh.folderPath == '/'.join(tmp_file.split('/')[:-1])

    def test_get(self, tmp_file):
        file_fh = FH(tmp_file, create=False)
        assert file_fh.get("pdf") == f"{tmp_file.split('.')[0]}.pdf"

    def test_file_notFound(self):
        with pytest.raises(FileNotFoundError):
            FH("NotExistingFile", create=False)

    def test_file_hd_with(self, tmp_file):
        with FH(tmp_file, create=False) as fh:
            assert isinstance(fh, FH)
            assert fh.path == tmp_file

    def test_equal(self, tmp_file, second_tmp_file):
        file1 = FH(tmp_file, create=False)
        file2 = FH(second_tmp_file, create=False)
        third_file = copy.deepcopy(tmp_file)
        file3 = FH(third_file, create=False)
        assert file1 == file3
        assert file1 != file2

    def test_autoDetect(self, conf_dir):
        files = FH.detect(DH(conf_dir, create=False))
        for file in files:
            assert isinstance(file, FH)
            assert FH.exists(file.path)

    def test_str(self, tmp_file):
        file_fh = FH(tmp_file, create=False)
        assert str(file_fh) == f"File handler: {str(tmp_file)}"

