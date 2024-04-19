# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import copy

from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.directories import DirectoryHandler as DH

class TestFileHandler:

    def test_file(self, test_file):
        file_fh = FH(test_file, create=False)
        assert isinstance(file_fh, FH)
        assert file_fh.path == test_file

    def test_folderPath(self, test_file):
        file_fh = FH(test_file, create=False)
        assert file_fh.folderPath == '/'.join(test_file.split('/')[:-1])

    def test_get(self, test_file):
        file_fh = FH(test_file, create=False)
        assert file_fh.get("pdf") == f"{test_file.split('.')[0]}.pdf"

    def test_file_notFound(self, test_file):
        with pytest.raises(FileNotFoundError):
            FH("NotExistingFile", create=False)

    def test_file_hd_with(self, test_file):
        with FH(test_file, create=False) as fh:
            assert isinstance(fh, FH)
            assert fh.path == test_file

    def test_equal(self, test_file, second_test_file):
        file1 = FH(test_file, create=False)
        file2 = FH(second_test_file, create=False)
        third_file = copy.deepcopy(test_file)
        file3 = FH(third_file, create=False)
        assert file1 == file3
        assert file1 != file2

    def test_autoDetect(self, detect_test_folder):
        conf, graph, events = FH.detect(DH(detect_test_folder, create=False))
        assert isinstance(conf, FH)
        assert isinstance(events, FH)
        assert isinstance(graph, FH)
        assert FH.exists(conf.path)
        assert FH.exists(events.path)
        assert FH.exists(graph.path)

    def test_autoDetect_error(self, detect_test_folder_error):
        with pytest.raises(FileNotFoundError):
            FH.detect(DH(detect_test_folder_error, create=False))
