# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
