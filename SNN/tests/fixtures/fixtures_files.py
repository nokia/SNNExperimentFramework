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

"""
log fixtures
============

Fixtures used for the lof tests
"""

import os
import pytest

@pytest.fixture
def tmp_files_path(tmp_path):
    p = tmp_path / "test_files"
    if not os.path.exists(p):
        p.mkdir()
    return p

@pytest.fixture
def tmp_file(tmp_files_path):
    p_init = tmp_files_path / "test_file.txt"
    p_init.touch()
    return str(p_init)

@pytest.fixture
def second_tmp_file(tmp_files_path):
    p_init = tmp_files_path / "second_test_file.txt"
    p_init.touch()
    return str(p_init)
