from csv import excel_tab
from re import I
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as mplt
import matplotlib.ticker as mtkr
import ptitprince as pt
from itertools import product
import subprocess

from SNN2.src.io.directories import DirectoryHandler as DH
from SNN2.src.plot.plotter import plotter as plt

from typing import Dict, List

parser = argparse.ArgumentParser(usage="usage: plot.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    output: DH = DH(options.output_folder)

    evaluation_df = pd.read_csv(input)
    print(evaluation_df)
    accuracy_eval_df = evaluation_df[(evaluation_df["Statistic"] == "MNO_Accuracy") | (evaluation_df["Statistic"] == "OTT_Accuracy")]
    MNO_cm_labels = [f"{s}_{o}_{v}" for s, o, v in product(["MNO"], ["0", "1", "2"], ["TP", "FP", "TN", "FN", "U"])]
    OTT_cm_labels = [f"{s}_{o}_{v}" for s, o, v in product(["OTT"], ["0", "1", "2"], ["TP", "FP", "TN", "FN", "U"])]
    MNO_expected_cm_labels = [f"{s}_{o}_expected_{v}" for s, o, v in product(["MNO"], ["0", "1", "2"], ["P", "N"])]
    OTT_expected_cm_labels = [f"{s}_{o}_expected_{v}" for s, o, v in product(["OTT"], ["0", "1", "2"], ["P", "N"])]
    MNO_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(MNO_cm_labels)].copy()
    OTT_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(OTT_cm_labels)].copy()
    MNO_expected_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(MNO_expected_cm_labels)].copy()
    OTT_expected_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(OTT_expected_cm_labels)].copy()

    def separate_org_cm(df: pd.DataFrame) -> pd.DataFrame:
        splt = df['Statistic'].str.split('_(.*?)_', expand=True)
        df = pd.concat([df, splt], axis=1)
        return df

    def normalize_cm(df, exp_df) -> pd.DataFrame:
        for i in set(exp_df.index):
            sub_exp_df = exp_df.loc[i]
            p = sub_exp_df[sub_exp_df["Label"] == "expected_P"].values[0][0]
            n = sub_exp_df[sub_exp_df["Label"] == "expected_N"].values[0][0]
            sub_df = df.loc[i].copy()
            p_mask = (sub_df["CM"] == "TP") | (sub_df["CM"] == "FN")
            n_mask = (sub_df["CM"] == "TN") | (sub_df["CM"] == "FP")
            u_mask = sub_df["CM"] == "U"
            sub_df.loc[p_mask, "Value"] = (sub_df.loc[p_mask]["Value"] / p)
            sub_df.loc[n_mask, "Value"] = (sub_df.loc[n_mask]["Value"] / n)
            sub_df.loc[u_mask, "Value"] = (sub_df.loc[u_mask]["Value"] / (p+n))
            df.loc[i] = sub_df.loc[i]
        df.fillna(0, inplace=True)
        return df

    def cm_pp(df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
        df = separate_org_cm(df)
        exp_df = separate_org_cm(expected_df)
        df.rename({'Origin': "Approach", 0: 'Side', 1: 'Origin', 2: 'CM'}, axis=1, inplace=True)
        exp_df.rename({'Origin': "Approach", 0: 'Side', 1: 'Origin', 2: 'Label'}, axis=1, inplace=True)
        df.drop(["Statistic", "Side"], axis=1, inplace=True)
        exp_df.drop(["Statistic", "Side"], axis=1, inplace=True)
        df = df.replace({'Origin': "0"}, value="Positive")
        df = df.replace({'Origin': "1"}, value="Negative")
        df = df.replace({'Origin': "2"}, value="Difficult")
        # df = df.set_index(["Experiment", "Approach", "Origin"])
        exp_df = exp_df.replace({'Origin': "0"}, value="Positive")
        exp_df = exp_df.replace({'Origin': "1"}, value="Negative")
        exp_df = exp_df.replace({'Origin': "2"}, value="Difficult")
        # exp_df = exp_df.set_index(["Experiment", "Approach", "Origin"])
        # df = normalize_cm(df, exp_df)
        return df

    def get_cm_stats(df, o, s):
        pcr = [.25, .5, .75]
        tp = df[df["CM"] == "TP"]["Value"].describe(percentiles=pcr)
        fp = df[df["CM"] == "FP"]["Value"].describe(percentiles=pcr)
        tn = df[df["CM"] == "TN"]["Value"].describe(percentiles=pcr)
        fn = df[df["CM"] == "FN"]["Value"].describe(percentiles=pcr)
        u = df[df["CM"] == "U"]["Value"].describe(percentiles=pcr)
        stat = pd.concat([tp, fp, tn, fn, u], axis=1)
        stat.columns = ["TP", "FP", "TN", "FN", "U"]
        stat = stat.T.copy()
        stat = stat.round(3)
        stat["Statistic"] = ["TP", "FP", "TN", "FN", "U"]
        stat["Side"] = s
        stat["Origin"] = o
        stat.drop(["count"], axis=1, inplace=True)
        stat.reset_index(inplace=True, drop=True)
        return stat

    def calculate_stats(df, s) -> pd.DataFrame:
        df_stats = pd.DataFrame()
        for org in df["Origin"].unique():
            tmp_tmp_df = df[df["Origin"] == org]
            tmp_tmp_st = get_cm_stats(tmp_tmp_df, org, s)
            df_stats = pd.concat([df_stats, tmp_tmp_st])
        df_stats.drop(["min", "max"], axis=1, inplace=True)
        df_stats["Mean (std)"] = df_stats["mean"].astype(str) + " (" + df_stats["std"].astype(str) + ")"
        df_stats.drop(["mean", "std"], axis=1, inplace=True)
        df_stats.set_index(["Side", "Origin", "Statistic"], inplace=True)
        cols = df_stats.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_stats = df_stats[cols]
        return df_stats

    def define_cm(df, expected_df, s) -> None:
        cm = cm_pp(df, expected_df)
        cm = cm.reset_index(drop=False)
        # cm.drop(["Experiment"], axis=1, inplace=True)
        cm_stats = calculate_stats(cm, s)
        cm_stats.to_csv(f"{output.path}/{s}_cm.csv")
        cm_stats.rename({'25%': "25\\%", "50%": '50\\%', "75%": '75\\%'}, axis=1, inplace=True)
        cm_stats[["25\\%","50\\%","75\\%"]] = cm_stats[["25\\%","50\\%","75\\%"]].round(3)
        cm_stats[["25\\%","50\\%","75\\%"]] = cm_stats[["25\\%","50\\%","75\\%"]].astype(str)

        tex_table = cm_stats.style.to_latex(column_format="|l|l|l|l|l|r|r|r|",
                                            position_float="centering",
                                            multicol_align="|c|",
                                            hrules=True)

        txt = f"""\\documentclass{{article}}
\\usepackage{{multirow}}
\\usepackage{{booktabs}}

\\title{{tmp}}
\\author{{Mattia Milani}}
\\date{{February 2023}}

\\begin{{document}}

{tex_table}

\\end{{document}}
"""
        with open(f"{output.path}/{s}_cm.tex", 'w+') as f:
            f.write(txt)
        proc = subprocess.Popen(['pdflatex', f"-output-directory={output.path}",f"{output.path}/{s}_cm.tex"])
        proc.communicate()


    define_cm(MNO_cm_eval_df, MNO_expected_cm_eval_df, "MNO")
    define_cm(OTT_cm_eval_df, OTT_expected_cm_eval_df, "OTT")


if __name__ == "__main__":
    main()
