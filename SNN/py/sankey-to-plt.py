from re import I
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as mplt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
import matplotlib.ticker as mtkr
import plotly.graph_objects as go
from plotly.offline import plot, iplot
from webcolors import hex_to_rgb
import matplotlib.patches as mpatches

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
    mplt.rc('font', size=50)
    # mplt.style.use(['science','ieee'])

    im = mplt.imread(input)
    fig = mplt.figure(figsize=(15,9))
    ax = fig.add_subplot()
    ax.imshow(im)
    ax.grid(False)
    ax.get_yaxis().set_visible(False)

    colors= ["#4daf4a", "#377eb8","#e41a1c"]
    b_patch = mpatches.Patch(color=colors[0], label=r"$\hat{U}_{good}$")
    g_patch = mpatches.Patch(color=colors[1], label=r"$\hat{U}_{bad}$")
    u_patch = mpatches.Patch(color=colors[2], label=r"$\hat{U}_{undecided}$")
    ax.legend(handles=[g_patch, b_patch, u_patch], ncol=3, loc="upper center", prop={'size': 34})
    ax.set_xticks([100, 840, 1680, 2500, 3330, 4150, 5000, 5820, 6660, 7500],
                  labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    ax.tick_params(axis="x", labelsize=38)
    mplt.xlabel("SL Iteration", fontsize=38)

    mplt.savefig(f"{output.path}/sankey_plt.png", dpi=400)
    mplt.savefig(f"{output.path}/sankey_plt.pdf", dpi=400)

if __name__ == "__main__":
    main()
