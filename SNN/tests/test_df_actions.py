# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


from numpy import NaN
import pytest
import pandas as pd

from SNN2.src.actions.dataFrame import load, remove_over_threshold
from SNN2.src.actions.dataFrame import write
from SNN2.src.actions.dataFrame import dropColumns
from SNN2.src.actions.dataFrame import dropOutliers
from SNN2.src.actions.dataFrame import removeNaN

class TestDFActions():

    def test_load(self, df, df_csv_file):
        loaded_df = load(df_csv_file, index_col=0)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_write(self, df, df_csv_file, df_folder_path):
        write_file = str(df_folder_path / "written_df.csv")
        write(write_file, df=df)
        red_df = load(write_file, index_col=0)
        copy_df = load(df_csv_file, index_col=0)
        pd.testing.assert_frame_equal(red_df, copy_df)

    def test_write_error(self, df_folder_path):
        write_file = str(df_folder_path / "written_df.csv")
        with pytest.raises(Exception):
            write(write_file)

    def test_dropColumns(self, df):
        tmp_df = df.copy()
        df = dropColumns(['A', 'B'], df=df)
        pd.testing.assert_frame_equal(df,
                tmp_df.drop(['A', 'B'], axis=1))

    def test_dropOutliers(self, exp_df):
        result = dropOutliers(df=exp_df, threshold=80.)
        pd.testing.assert_frame_equal(result,
                exp_df[exp_df['exp_id'] != 2])

    def test_removeNaN(self, exp_df_nan) -> None:
        result = removeNaN(df=exp_df_nan)
        result_df = pd.DataFrame({
            'problem': ["good", "good", "good"],
            'exp_id': [1, 2, 2],
            'vmaf': [100., 50., 50]
        })
        pd.testing.assert_frame_equal(result, result_df)

    def test_remove_over_threshold(self, exp_df_nan):
        result: pd.DataFrame = remove_over_threshold(df=exp_df_nan,
                                                     column="vmaf",
                                                     threshold=10.)
        result.reset_index(drop=True, inplace=True)
        result_df = pd.DataFrame({
            'problem': ["bad", "bad"],
            'exp_id': [3, 3],
            'vmaf': [NaN, 10.]
        })
        pd.testing.assert_frame_equal(result, result_df)
