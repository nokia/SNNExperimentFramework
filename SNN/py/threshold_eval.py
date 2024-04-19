from re import I
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as mplt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
import matplotlib.ticker as mtkr

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
    sns.set_theme(context="paper")
    # mplt.style.use(['science','ieee'])

    colors = ["#4daf4a", "#377eb8", "#e41a1c"]
    cm = sns.color_palette(colors)
    # font_file = '/home/mattia/src/SNNFrameWork/Fonts/helvetica.ttf'
    # fm.fontManager.addfont(font_file)
    # font_manager.fontManager.ttflist.extend(font_list)
    custom_params = {'figure.figsize':(8,4)}
    sns.set(font_scale=1.65)
    sns.set_theme(rc=custom_params)

    df = pd.read_csv(input)

    for kpi in df["threshold_name"].unique():
        kpi_df = df[df["threshold_name"] == kpi]
        p = plt(kpi_df, format=["pdf", "png"])
        p.sns_set(font_scale=1.8)
        p.sns_set_api(sns.set_style, "white")
        p("lineStack", x="threshold_value", y="Value", hue="Statistic", normalize=True, colors=cm)
        p.set(xlabel=r"PDR for $\mathcal{B}_k$",
              ylabel="Set distribution [%]")
        mplt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0', '25', '50', '75', '100'])
        p.set_legend(loc="lower center", x=0.5, y=0.6, ncol=3, title="Dataset", labels=[r"$\hat{G}$", r"$\hat{B}$", r"$\hat{U}$"])
        p.save(f"{output.path}/datasetDistributionVSThreshold-{kpi}.pdf")



if __name__ == "__main__":
    main()
