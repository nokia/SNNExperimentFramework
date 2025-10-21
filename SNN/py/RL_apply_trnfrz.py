# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import seaborn as sns
import re
import argparse
import numpy as np
import pandas as pd
from SNN2.src.io.files import FileHandler as FH

from SNN2.src.plot.plotter import plotter as plt

from typing import Callable, Dict, List, Optional

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-n", "--delim", dest="delimiter", default=10, type=int,
                    action="store", help="How many episodes between Train and Frz")
parser.add_argument("-c", "--column", dest="column", default="Experiment", type=str,
                    action="store", help="Column where to compute the correct label")

def main():
    options = parser.parse_args()
    input: FH = FH(options.csv_file, create=False)
    dim: int = options.delimiter
    clm: str = options.column

    print(input.path)
    df = pd.read_csv(input.path)

    phase_col = None

    if dim == 0:
        phase_col= "NO_RL"
    else:
        for exp in df["Experiment"].unique():
            ep_col = df[df["Experiment"] == exp][clm].values
            trn_ep = max(ep_col) - dim
            tmp_phase_col = np.where(ep_col <= trn_ep, "RL_Training", "RL_Exploitation")
            if phase_col is None:
                phase_col = tmp_phase_col
            else:
                phase_col = np.append(phase_col, tmp_phase_col)

    df["Phase"] = phase_col
    columns = list(df.columns)
    columns.remove("Phase")
    columns.insert(1, "Phase")
    df = df[columns]
    df.to_csv(input.path,index=False)


if __name__ == "__main__":
    main()
