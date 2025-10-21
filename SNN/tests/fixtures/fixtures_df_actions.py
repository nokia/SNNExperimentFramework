# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
