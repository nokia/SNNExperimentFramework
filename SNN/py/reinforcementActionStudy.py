# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import seaborn as sns
import re
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from SNN2.src.io.files import FileHandler as FH

from SNN2.src.plot.plotter import plotter as plt

from typing import Any, Callable, Dict, List, Optional

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="",
                    action="store", help="Define the appendix")

def save(obj: Any, fp: str) -> None:
    if os.path.exists(fp):
        raise FileExistsError(f"{fp} already exists, pickle creation aborted")
    with open(fp, 'wb') as file:
        pkl.dump(object, file, protocol=pkl.HIGHEST_PROTOCOL)

def load(fp: str) -> Any:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} Not found, pickle loading aborted")
    with open(fp, 'rb') as file:
        obj = pkl.load(file)
        return obj

def state_action_dict(states: np.ndarray, actions: np.ndarray) -> Dict[str, Dict[str, int]]:
    res = {}
    for state, action in zip(states, actions):
        def prune(s):
            np.set_printoptions(suppress=True)
            return str(s)
        state = prune(state)
        if state not in res:
            res[state] = {}
            res[state]["0"] = 0
            res[state]["1"] = 0
        res[state][str(action)] += 1
    return res

def main():
    options = parser.parse_args()
    input: FH = FH(options.csv_file, create=False)
    output: str = options.output_folder
    appendix: str = options.appendix

    df = pd.read_csv(input.path)

    def toNpArray(s):
        res = re.sub(' +', ' ', s)
        res = re.sub(' ]', ']', res)
        res = res.replace('[', '').replace(']', '').replace(',', '').split(' ')
        res = np.array(res)
        res = res.astype(np.float16)
        res[0] = np.around(res, decimals=1)[0]
        return res

    if not os.path.exists(f"{output}/tmp_df_state_conversion{appendix}.pkl"):
        df["State"] = df["State"].apply(toNpArray)
        df.to_pickle(f"{output}/tmp_df_state_conversion{appendix}.pkl")
    else:
        df = pd.read_pickle(f"{output}/tmp_df_state_conversion{appendix}.pkl")

    if not os.path.exists(f"{output}/tmp_df_from_dict{appendix}.pkl"):
        s_a_dict = state_action_dict(df["State"].values, df["Value"].values)
        df_melt = pd.json_normalize(s_a_dict, sep='>>').melt()
        df_final = df_melt['variable'].str.split('>>', expand=True)
        df_final.columns = [f'col{name}' for name in df_final.columns]
        df_final['value'] = df_melt['value'].values
        df_final.to_pickle(f"{output}/tmp_df_from_dict{appendix}.pkl")
    else:
        df_final = pd.read_pickle(f"{output}/tmp_df_from_dict{appendix}.pkl")

    df_final.drop(["col0"], axis=1, inplace=True)

    print(df_final)
    print(df_final[df_final["value"] > 1].describe())
    print(df_final[df_final["value"] == 0])
    print(df_final[(df_final["value"] == 0) & (df_final["col2"] == "0")])
    print(df_final[(df_final["value"] == 0) & (df_final["col2"] == "1")])

if __name__ == "__main__":
    main()
