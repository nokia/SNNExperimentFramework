import os
import math
from re import I
import sys
import argparse
from typing import List
import numpy as np
import pandas as pd
import scipy.stats as stats
import more_itertools as mit

parser = argparse.ArgumentParser(usage="usage: nsVmafMerge.py [options]",
                                 description="Use the script to generate statistic csvs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/tshark.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-v", "--vmaf", dest="vmaf_file", default="result/vmaf.csv",
                    action="store", help="define the vmaf input file")
parser.add_argument("-o", "--output", dest="output", default="stats.csv",
                    action="store", help="Define the file where to store the stats")

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    vmaf: str = options.vmaf_file
    output: str = options.output

    ndata = pd.read_csv(input)
    vdata = pd.read_csv(vmaf)

    vdata = vdata[["vmaf"]]

    fps = math.ceil(len(vdata)/len(ndata))
    s = np.repeat(np.arange(1, len(ndata)+1), fps)
    s = s[:len(vdata)]
    vdata["second"] = s
    vdata = vdata.groupby(['second']).mean().reset_index(drop=False)

    ndata["vmaf"] = vdata["vmaf"]

    ndata.to_csv(output)

if __name__ == "__main__":
    main()

