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

from numpy import NaN
import pytest
import pandas as pd

@pytest.fixture
def df():
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

@pytest.fixture
def df_folder_path(tmp_path):
    p = tmp_path / "df"
    p.mkdir()
    return p

@pytest.fixture
def df_csv_file(df, df_folder_path):
    p_init = df_folder_path / "df.csv"
    df.to_csv(p_init)
    return p_init

@pytest.fixture
def exp_df():
    return pd.DataFrame({
        'problem': ["good", "good", "bad"],
        'exp_id': [1, 2, 3],
        'vmaf': [100, 50, 10]
    })

@pytest.fixture
def exp_df_nan() -> pd.DataFrame:
    return pd.DataFrame({
        'problem': ["good", "good", "good", "bad", "bad"],
        'exp_id': [1, 2, 2, 3, 3],
        'vmaf': [100, 50, 50, NaN, 10]
    })
