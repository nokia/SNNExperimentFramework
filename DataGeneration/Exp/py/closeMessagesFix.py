import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import more_itertools as mit

columns = ["frame", "arrival_time", "diff_time", "frame_len", "id", "PID1",
           "PID2", "PID3", "PID4", "PID5", "PID6", "PID7", "cc1", "cc2", "cc3",
           "cc4", "cc5", "cc6", "cc7"]

parser = argparse.ArgumentParser(usage="usage: closeMessagesFix.py [options]",
                                 description="Use the script to generate the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/tshark.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output", default="stats.csv",
                    action="store", help="Define the file where to store the stats")

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    output: str = options.output
    delta: float = 0.000005
    # pd.set_option('display.precision', 7)

    df = pd.read_csv(input)
    clean_df = df.copy()
    clean_df = clean_df[(clean_df["diff_time"] > delta) | (clean_df["frame"] == 1)]
    # print(clean_df)

    # print(len(df[df["diff_time"] <= delta]))

    clean_df.to_csv(output, index=False, float_format='%f')

if __name__ == "__main__":
    main()

