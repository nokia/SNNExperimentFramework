import os
from re import I
import sys
import argparse
from typing import List
import numpy as np
import pandas as pd
import scipy.stats as stats
import more_itertools as mit

parser = argparse.ArgumentParser(usage="usage: ns2data.py [options]",
                                 description="Use the script to generate statistic csvs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", nargs='+', dest="csv_file", default="result/tshark.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output", default="stats.csv",
                    action="store", help="Define the file where to store the stats")
parser.add_argument("-d", "--delta", dest="delta", default=0, type=float,
                    action="store", help="Define the header delta")
parser.add_argument("-L", "--LUnknown", dest="boh", default=False,
                    action="store_true", help="Retro compatible")

def main():
    options = parser.parse_args()
    inputs: List[str] = options.csv_file
    output: str = options.output
    delta: float = options.delta

    assert len(inputs) == 2

    relevant_columns = ["packets", "bytes"]
    df1 = pd.read_csv(inputs[0])
    df2 = pd.read_csv(inputs[1])
    df1[relevant_columns] = df1[relevant_columns].astype(float)
    df2[relevant_columns] = df2[relevant_columns].astype(float)

    relevant_df1 = df1[relevant_columns].copy()
    relevant_df2 = df2[relevant_columns].copy()

    delta = np.repeat(np.array([delta]), len(relevant_df2))
    relevant_df2["bytes"] -= delta*relevant_df2["packets"]

    diff = relevant_df1 - relevant_df2

    diff["packets"] /= relevant_df1["packets"]
    diff["bytes"] /= relevant_df1["bytes"]

    df2["packets"] = diff["packets"]
    df2["bytes"] = diff["bytes"]
    new_columns = ["packet_drop_rate", "byte_drop_rate", "avg_timeDelta", "std_timeDelta", "skw_timeDelta", "kur_timeDelta", "second"]
    df2.columns = new_columns
    new_columns = ["second"] + new_columns[:-1]
    df2 = df2[new_columns]
    df2 = df2.round(4)
    df2.to_csv(output, index=False)


if __name__ == "__main__":
    main()

