from re import I
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import re
import matplotlib.pyplot as mplt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
import matplotlib.ticker as mtkr
import ptitprince as pt
from itertools import product
import subprocess
import scienceplots
from matplotlib import rcParams
import pickle as pkl
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

from SNN2.src.io.directories import DirectoryHandler as DH
from SNN2.src.plot.plotter import plotter as plt

from typing import Any, Dict, List

parser = argparse.ArgumentParser(usage="usage: plot.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--folder", dest="pkl_folder", default="result/statistics.csv",
                    action="store", help="define the pkl input folder")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")

def load(file: str) -> Any:
    with open(file, 'rb') as f:
        return pkl.load(f)

def matrix2df(matrix: tf.Tensor,
              norm: bool = True) -> pd.DataFrame:
    idx = range(matrix.shape[0])
    cols = range(matrix.shape[1])
    df = pd.DataFrame(matrix.numpy(), index=idx, columns=cols)
    return df

def get_3d_matrix(file_list: List[str]) -> tf.Tensor:
    matrixes = None
    for i, file in enumerate(file_list):
        matrix = load(file)
        matrixes = tf.expand_dims(matrix, 0) if matrixes is None else tf.concat([matrixes, tf.expand_dims(matrix, 0)], axis=0)
    return matrixes

def normalize_3d_matrix(matrixes: tf.Tensor) -> tf.Tensor:
    no_diag_matrix = tf.where(tf.equal(matrixes, 1.0), 0.0, matrixes)
    class_min = tf.reduce_min(tf.reduce_min(no_diag_matrix, axis=2, keepdims=True), axis=0, keepdims=True)
    class_max = tf.reduce_max(tf.reduce_max(no_diag_matrix, axis=2, keepdims=True), axis=0, keepdims=True)
    div = tf.math.subtract(class_max, class_min)
    matrixes_norm = tf.math.subtract(no_diag_matrix, class_min)
    matrixes_norm = tf.math.divide(matrixes_norm, div)
    # matrixes_norm = tf.math.multiply(matrixes_norm, 2.0)
    matrixes_norm = tf.math.subtract(matrixes_norm, 1.0)
    matrixes_norm = tf.where(tf.equal(matrixes, 0.0), 0.0, matrixes_norm)
    matrixes_norm = tf.where(tf.equal(matrixes, 1.0), 1.0, matrixes_norm)
    return matrixes_norm

def plot_matrix_list(file_list: List[str],
                     output: DH,
                     norm: bool = True,
                     prefix: str = "corr_matrix_ep",
                     mult: int = 10) -> None:
    matrixes = get_3d_matrix(file_list)
    if norm:
        matrixes = normalize_3d_matrix(matrixes)

    # l_i = [1, 3, 69, 118, 130, 131, 43, 113]
    # l_j = [36, 10, 82, 2, 3, 6, 66, 35]
    # l_i = range(112, 118)
    # for i in l_i:
    #     print(i)
    #     line = matrixes[0, i, :]
    #     print(tf.gather(line, tf.where(line < -0.5)[:, 0]))
    #     print(tf.gather(tf.range(line.shape[0]), tf.where(line < -0.5)[:, 0]))
    # raise Exception
    # for i, j in zip(l_i, l_j):
    #     couple_evo = matrixes[:, i, j]
    #     print(f"{i}-{j}: {couple_evo}")
    # raise Exception

    ids = tf.where(matrixes[0, :, :] < -0.01)
    print(ids)
    for i in range(141):
        tmp_ids = tf.gather(ids, tf.where(ids[:, 0] == i)[:, 0])
        if len(tmp_ids) <= 1:
            pass
        elif i != 75:
            pass
        else:
            # best_ids = tf.gather(ids, [2, 43, 48, 50, 83])

            tmp_ids = tf.gather(tmp_ids, [1, 2, 3])
            new_m = None
            for id in tmp_ids:
                tmp_i, j = id
                print(tmp_i, j)
                new_m = tf.expand_dims(matrixes[:, tmp_i, j], 0) if new_m is None else tf.concat([new_m, tf.expand_dims(matrixes[:, i, j], 0)], axis=0)

            k = tf.range(1, 41)
            idx = tf.where(k == 1, True, False)
            idx = tf.where(k%5 == 0, True, idx)
            idx = tf.expand_dims(idx, 0)
            idx = tf.concat([idx, idx, idx], axis=0)
            idx = tf.where(idx)
            sub_m = None
            for id in idx:
                k, j = id
                sub_m = tf.expand_dims(new_m[k, j], 0) if sub_m is None else tf.concat([sub_m, tf.expand_dims(new_m[k, j], 0)], axis=0)

            sub_m = tf.reshape(new_m, (new_m.shape[0], -1))
            print(sub_m)

            df_m = matrix2df(new_m, norm=norm)
            print(df_m)
            p = plt(df_m, format=["pdf", "png"])
            p.sns_set(font_scale=2.0)
            p.sns_set_api(sns.set_style, "white")
            cmap = sns.color_palette("viridis", as_cmap=True)
            num_colors = 20
            colors = [cmap(i / num_colors) for i in range(2, num_colors)]
            cmap = LinearSegmentedColormap.from_list('', colors, num_colors)
            p("heatmap", annot=False, vmin=-1, vmax=0.0, cmap=cmap, linewidths=0.0, xticklabels=False, yticklabels=False, cbar_kws = dict(location="bottom", ticks=[-1.0, -0.75, -0.5, -0.25, 0.0]))

            ax = p.plot

            # Get the images on an axis
            im = ax.collections

            # Assume colorbar was plotted last one plotted last
            cb = im[-1].colorbar
            cb.ax.set_xticklabels(['0.0', '-0.25', '-0.5', '-0.75', '-1.0'])

            p.set(ylabel=r"$C_{ij}$", xlabel="Time")
            out = f"{output.path}/best_examples_i{i}.pdf"
            print(out)
            p.save(f"{output.path}/best_examples_i{i}.pdf")

    raise Exception
    # best_ids = tf.gather(ids, [1, 2, 7, 25, 36, 43, 48, 50, 74, 83])
    best_ids = tf.gather(ids, [43, 48, 83])
    # best_ids = tf.gather(ids, [2, 43, 48, 50, 83])
    new_m = None
    for id in best_ids:
        i, j = id
        print(i, j)
        new_m = tf.expand_dims(matrixes[:, i, j], 0) if new_m is None else tf.concat([new_m, tf.expand_dims(matrixes[:, i, j], 0)], axis=0)

    i = tf.range(1, 41)
    idx = tf.where(i == 1, True, False)
    idx = tf.where(i%5 == 0, True, idx)
    idx = tf.expand_dims(idx, 0)
    idx = tf.concat([idx, idx, idx], axis=0)
    idx = tf.where(idx)
    sub_m = None
    for id in idx:
        i, j = id
        sub_m = tf.expand_dims(new_m[i, j], 0) if sub_m is None else tf.concat([sub_m, tf.expand_dims(new_m[i, j], 0)], axis=0)

    sub_m = tf.reshape(sub_m, (new_m.shape[0], -1))

    df_m = matrix2df(sub_m, norm=norm)
    p = plt(df_m, format=["pdf", "png"])
    p.sns_set(font_scale=2.0)
    p.sns_set_api(sns.set_style, "white")
    cmap = sns.color_palette("viridis", as_cmap=True)
    num_colors = 20
    colors = [cmap(i / num_colors) for i in range(2, num_colors)]
    cmap = LinearSegmentedColormap.from_list('', colors, num_colors)
    p("heatmap", annot=False, vmin=-1, vmax=0.0, cmap=cmap, linewidths=0.0, xticklabels=False, yticklabels=False, cbar_kws = dict(location="bottom", ticks=[-1.0, -0.75, -0.5, -0.25, 0.0]))

    ax = p.plot

    # Get the images on an axis
    im = ax.collections

    # Assume colorbar was plotted last one plotted last
    cb = im[-1].colorbar
    cb.ax.set_xticklabels(['0.0', '-0.25', '-0.5', '-0.75', '-1.0'])

    p.set(ylabel=r"$C_{ij}$", xlabel="Time")
    p.save(f"{output.path}/best_examples.pdf")
    raise Exception

    print(new_m)
    print(new_m.shape)
    raise Exception

    i_range = (112, 125)
    for i in range(len(file_list)):
        matrix = matrixes[i, i_range[0]:i_range[1], :]
        matrix = tf.reshape(matrix, (i_range[1]-i_range[0], 141))
        df_m = matrix2df(matrix, norm=norm)
        p = plt(df_m, format=["pdf", "png"])
        p.sns_set(font_scale=2.0)
        p.sns_set_api(sns.set_style, "white")
        cmap = sns.color_palette("vlag", as_cmap=True)
        num_colors = 20
        colors = [cmap(i / num_colors) for i in range(2, num_colors)]
        cmap = LinearSegmentedColormap.from_list('', colors, num_colors)

        p("heatmap", annot=False, vmin=-1, vmax=1.0, cmap=cmap, square=True, linewidths=0.0, xticklabels=False, yticklabels=False, cbar_kws = dict(location="bottom"))
        p.set(xlabel=r"Class $j$", ylabel=r"Class $i$")
        p.save(f"{output.path}/{prefix}{i*mult}.pdf")
        raise Exception


def main():
    options = parser.parse_args()
    input: DH = DH(options.pkl_folder, create=False)
    output: DH = DH(options.output_folder)

    sns.set_theme(context="paper")
    # mplt.style.use(['science','ieee'])

    # colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    colors = ["#e41a1c", "#377eb8"]
    cm = sns.color_palette(colors)
    font_file = '/home/mattia/src/SNNFrameWork/Fonts/helvetica.ttf'
    fm.fontManager.addfont(font_file)
    # font_manager.fontManager.ttflist.extend(font_list)
    custom_params = {'figure.figsize':(8,8), "font.size": 44}
    sns.set_theme(rc=custom_params)
    rcParams['figure.figsize'] = 8,8
    rcParams['font.size'] = 44
    rcParams['axes.titlepad'] = 20

    file_name = ".*/corr_matrix_iteration_.*"
    r = re.compile(file_name)
    corr_files = list(filter(r.match, input.files)) # Read Note
    plot_matrix_list(corr_files, output)

    return 0
    file_name = ".*/phi_matrix_iteration_.*"
    r = re.compile(file_name)
    corr_files = list(filter(r.match, input.files)) # Read Note
    plot_matrix_list(corr_files, output, prefix="phi_matrix_ep", norm=False)

if __name__ == "__main__":
    main()
