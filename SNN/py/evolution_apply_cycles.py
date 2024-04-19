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
parser.add_argument("-r", "--rl", dest="reinforce_cycle", default=2, type=int,
                    action="store", help="Define the number of cycles used during the RL_train phase")
parser.add_argument("-c", "--column", dest="column", default="Experiment", type=str,
                    action="store", help="Column where to compute the correct label")

def main():
    options = parser.parse_args()
    input: FH = FH(options.csv_file, create=False)
    dim: int = options.delimiter
    rl_dim: int = options.reinforce_cycle
    clm: str = options.column
    epoch_clm: str = "epoch"
    exp_clm: str = "Experiment"

    df = pd.read_csv(input.path)

    epoch_min_cycle_dim = 0
    epoch_max_cycle_dim = df[epoch_clm].max()+1
    phase_col = None

    evaluation = []
    cycle = []
    phase = None
    for exp in df[exp_clm].unique():
        tmp_df = df[df[exp_clm] == exp].copy()
        if dim == 0:
            phase = "NO_RL"
            num_cycles = len(tmp_df[tmp_df[epoch_clm] == epoch_min_cycle_dim])
            epochs = tmp_df[epoch_clm].values
            current_evaluation = -1
            for ep in epochs:
                if ep == 0:
                    current_evaluation += 1
                evaluation.append(current_evaluation)
                cycle.append(0)
        else:
            phase = [] if phase is None else phase
            num_cycles = len(tmp_df[tmp_df[epoch_clm] == epoch_min_cycle_dim])
            rl_traning_cycles = (num_cycles-dim)
            cycles_for_each_rl_trn = rl_dim
            rl_exploit_cycles = dim
            epochs = tmp_df[epoch_clm].values
            current_evaluation = -1
            current_phase = "RL_Training"
            for ep in epochs:
                if ep == 0:
                    current_evaluation += 1
                if current_evaluation == rl_traning_cycles:
                    current_phase = "RL_Exploitation"
                    current_evaluation = 0
                evaluation.append(current_evaluation)
                phase.append(current_phase)


    df["Evaluation"] = evaluation
    df["Phase"] = phase
    columns = list(df.columns)
    columns.remove("Evaluation")
    columns.remove("Phase")
    columns.insert(2, "Evaluation")
    columns.insert(1, "Phase")
    df = df[columns]
    df.to_csv(input.path,index=False)

if __name__ == "__main__":
    main()
