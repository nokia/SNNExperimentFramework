import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from SNN2.src.plot.plotter import plotter as plt

from typing import Dict, List

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")

def to_long(df, **kwargs):
    tmp_df = pd.wide_to_long(df, **kwargs)
    tmp_df.reset_index(inplace=True)
    return tmp_df

def flag_count_ecdf(df: pd.DataFrame, output: str) -> None:
    df = df[df["KPI"] == "Flag"]
    df = df.drop(["KPI"], axis=1)
    df = df.groupby(["SampleID", "Values"])["Values"].count().reset_index(name="Count")
    df = df.loc[df.groupby(["SampleID"])["Count"].idxmax()]

    df.columns = ["SampleID", "Class", "Count"]
    df.loc[df["Class"] == 0.0, "Class"] = "Good"
    df.loc[df["Class"] == 1.0, "Class"] = "Bad"
    df.loc[df["Class"] == 2.0, "Class"] = "Undecided"

    p = plt(df, format=["pdf", "png"])
    p.ecdf(x="Count", hue="Class")
    p.set(title=f"Number of time a decision has been taken",
          xlim=(0, None))
    p.save(f"{output}gray_prediction_ecdf.pdf")


def main():
    options = parser.parse_args()
    input: str = options.csv_file
    output: str = options.output_folder

    df = pd.read_csv(input)

    flag_count_ecdf(df, output)

if __name__ == "__main__":
    main()
