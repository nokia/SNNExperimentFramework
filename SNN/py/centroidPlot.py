import os
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.plot.plotter import plotter as plt

from typing import Dict, List, Any, Callable, Tuple

parser = argparse.ArgumentParser(usage="usage: centroidStudy.py [options]",
                                 description="Use the script to study samples vs centroids",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="dstFile", default=None,
                    action="store", help="define the distance embeddings file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="iteration0",
                    action="store", help="Define the appendix")

def main():
    options = parser.parse_args()
    dst_file: FH = FH(options.dstFile, create=False)
    output: str = options.output_folder
    appendix: str = options.appendix

    df = pd.read_csv(dst_file.path)

    p = plt(df, format=["pdf", "png"])
    p("violin", x="Class", y="Value", hue="Distances", split=True, scale="count")
    p.set(title=f"Sample distance from P and N centroids {appendix}",
          ylabel="Distance",
          xlabel="Sample Class")
    p.set_legend()
    p.save(f"{output}/sample_vs_centroid_distance_comparison{appendix}.pdf")


if __name__ == "__main__":
    main()
