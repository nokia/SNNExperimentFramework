# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Main module
===========

Main module of the project, use it to load all the different parts and
generate the required plots

"""

import argparse
import matplotlib.font_manager
import pandas as pd
from matplotlib import rc
import seaborn as sns
from SNN2.src.io.files import FileHandler as FH
from SNN2.src.io.directories import DirectoryHandler as DH
from SNN2.src.io.commandInterpolation import str_interpolate as si
from SNN2.src.plot.plotter import plotter as plt

parser = argparse.ArgumentParser(usage="usage: conext-plot.py [options]",
                      description="Generate the CoNEXT plots",
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", dest="input", default="statistics.csv",
                    action="store", help="define the input file")
parser.add_argument("-o", "--output", dest="output", default="plots-{!date}",
                    action="store", help="define the outputFolder")

def plot(df: pd.DataFrame, output_file: str, *args, **kwargs) -> None:
    p = plt(df, format=["pdf", "png"], palette=sns.color_palette(["#e41a1c", "#377eb8", "#4daf4a"]))
    order=["Bunny", "Bottle", "Scarlet"]
    p("line", *args, x="value", y="vmaf", hue="video", style="video", hue_order=order, style_order=order)
    p.set(**kwargs)
    p.save(output_file)

def main():
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    options = parser.parse_args()
    input: str = options.input
    output: str = options.output
    input_fh: FH = FH(input, create=False)
    output_dh: DH = DH(si(output))

    df: pd.DataFrame = pd.read_csv(input_fh.path)
    df.drop(["second", "packet_drop_rate", "byte_drop_rate", "avg_timeDelta", "std_timeDelta", "skw_timeDelta", "kur_timeDelta"], axis=1, inplace=True)
    df["video"] = df["video"].str.capitalize()
    df: pd.DataFrame = df.groupby(["exp_id", "video", "problem"]).mean().reset_index(drop=False)
    df.drop(["exp_id"], axis=1, inplace=True)

    drop_df = df[(df["problem"] == "drop") & (df["video"] != "football")]
    plot(drop_df, f"{output_dh.path}/drop.pdf", xlabel="Drop [\%]", ylabel="VMAF", xlim=(0.0, None))
    delay_df = df[(df["problem"] == "delay") & (df["video"] != "football")]
    plot(delay_df, f"{output_dh.path}/delay.pdf", xlabel="Delay [ms]", ylabel="VMAF", xlim=(0.0, 300))

if __name__ == "__main__":
	main()

