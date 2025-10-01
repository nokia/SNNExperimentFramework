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
test configuration module
=========================

This module contains the fixture for the test suite
"""

import pytest

from fixtures.fixtures_log import log_file
from fixtures.fixtures_log import dummy_log
from fixtures.fixtures_params import generic_param, callable_param, env_rng_param, env_tf_rng_param
from fixtures.fixtures_callable import dummy_callable
from fixtures.fixtures_dirs import conf_dir
from fixtures.fixtures_files import tmp_files_path
from fixtures.fixtures_files import tmp_file
from fixtures.fixtures_files import second_tmp_file
from fixtures.fixtures_df_actions import df, df_csv_file, df_folder_path, exp_df, exp_df_nan
from fixtures.fixtures_tf_cure import tf_sample_LSUN

@pytest.fixture
def test_fixture():
    return "hello"
