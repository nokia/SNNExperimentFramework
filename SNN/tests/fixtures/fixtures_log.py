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

import pytest

from SNN2.src.io.logger import LogHandler as LH

@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "log"
    p.mkdir()
    p_init = p / "log.log"
    p_init.touch()
    return p_init

@pytest.fixture
def dummy_log(log_file):
    return LH(log_file, 0)


