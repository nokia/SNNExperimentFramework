import os
import ast
import umap.umap_ as umap
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as mplt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle as pkl
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from SNN2.src.plot.plotter import plotter as plt

from typing import Dict, List, Any, Callable, Optional

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", nargs='+', dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("--origin", nargs='+', dest="origins", default=None,
                    action="store", help="define the origins input file")
parser.add_argument("--subselect", nargs='+', dest="subselect", default=None,
                    action="store", help="Define the subselection labels")
parser.add_argument("-l", "--labels", nargs='+', dest="labels", default="none",
                    action="store", help="define the labels")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="iteration0",
                    action="store", help="Define the appendix")
parser.add_argument("-i", "--intermidiate", dest="intermidiate", default="/tmp",
                    action="store", help="Define where to save and store intermidiate output as pkl file")

class Transform:
    """Transform.
    Class that applies general transformations to data
    """

    @classmethod
    def tsne2df(cls, emb: np.ndarray,
                dimensions: int = 2,
                columns: List[str] = ['x', 'y'],
                **kwargs) -> pd.DataFrame:
        """tsne2df.
        Transform an np.ndarray using TSNE.
        is possible to define the output number of dimensions
        and the kwargs that will be passed to the TSNE method of
        sklearn.manifold.

        Parameters
        ----------
        emb : np.ndarray
            emb
        dimensions : int
            dimensions
        columns : List[str]
            columns Define a label for each dimension of the output dataframe
        kwargs :
            kwargs

        Returns
        -------
        pd.DataFrame

        """
        tsne = TSNE(dimensions, **kwargs)
        tsne_result = tsne.fit_transform(emb)
        result = pd.DataFrame(tsne_result, columns=columns)
        return result

    @classmethod
    def umap2df(cls, emb: np.ndarray,
                dimension: int = 2,
                columns: List[str] = ['X', 'Y'],
                **kwargs) -> pd.DataFrame:
        reducer = umap.UMAP(n_neighbors=30,
                            min_dist=0.0)
        scaled_data = StandardScaler().fit_transform(emb)
        embedding = reducer.fit_transform(scaled_data)
        result = pd.DataFrame(embedding, columns=columns)
        return result

def load(file: str) -> Any:
    with open(file, 'rb') as f:
        return pkl.load(f)

def save(file: str, obj: Any) -> None:
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists")
    with open(file, 'wb') as f:
        pkl.dump(obj, f)

def reduction(inputs: List[str],
              labels: List[str],
              transformation_function: Callable = Transform.umap2df,
              origins: Optional[List[str]] = None,
              subselect: Optional[List[str]] = None,
              intermidiate: Optional[str] = None):
    if intermidiate is not None and os.path.exists(intermidiate):
        return load(intermidiate)
    embeddings = [load(input).numpy() for input in inputs]
    print(f"Length embeddings: {[emb.shape for emb in embeddings]}")
    apply_labels = np.concatenate([np.repeat(label, len(emb)) for label, emb in zip(labels, embeddings)], axis=0)

    if origins is not None and subselect is not None:
        for i, origin_subselect in enumerate(zip(origins, subselect)):
            origin, sub = origin_subselect
            origin = load(origin)
            label = labels[i]
            if sub == "Label":
                new_labels = tf.where(origin == 0, "Good2", "Bad2").numpy()
                new_labels = new_labels.astype('U13')
                apply_labels = np.delete(apply_labels, np.where(apply_labels == label)[0])
                apply_labels = np.concatenate([apply_labels, new_labels], axis=0)
                # print(apply_labels)
                # print(f"Sub: {sub}")
                # correct_origin = tf.where(origin == ast.literal_eval(sub))[:, 0]
                # print(f"Indexes: {correct_origin}")
                # embeddings[i] = np.take(embeddings[i], correct_origin.numpy(), axis=0)

    print(f"Length embeddings: {[emb.shape for emb in embeddings]}")

    embeddings = np.concatenate(embeddings, axis=0)
    emb_df = transformation_function(embeddings)
    emb_df["Data"] = apply_labels
    if intermidiate is not None:
        save(intermidiate, emb_df)
    return emb_df

def main():
    options = parser.parse_args()
    inputs: List[str] = options.csv_file
    labels: List[str] = options.labels
    origins: List[str] = options.origins
    subselect: List[str] = options.subselect
    output: str = options.output_folder
    appendix: str = options.appendix
    intermidiate: str = options.intermidiate

    assert len(inputs) == len(labels)

    intermidiate_file = f"{intermidiate}/dataframe_already_computed_{appendix}.pkl"
    df = reduction(inputs,
                   labels,
                   intermidiate=intermidiate_file,
                   origins=origins,
                   subselect=subselect)
    if df is None:
        raise Exception

    # Seaborn settings
    sns.set(font_scale=1.5)
    sns.set_style("white")
    sns.set_palette(sns.color_palette())
    legend_position = (0.5, -0.2)

    # p = plt(df, format=["pdf", "png"])
    # p("joint", x="X", y="Y", hue="Data", kind="kde", fill=True, thresh=0.05, alpha=.8, marginal_kws={'common_norm': False})
    # p.move_legend("upper center", obj=p.plot.ax_joint, bbox_to_anchor=legend_position, ncol=3)
    # p.save(f"{output}data_umap_representation_fill_{appendix}.pdf")

    # p("joint", x="X", y="Y", hue="Data", kind="kde", marginal_kws={'common_norm': False})
    # legend_position = (0.5, -0.2)
    # p.move_legend("upper center", obj=p.plot.ax_joint, bbox_to_anchor=legend_position, ncol=3)
    # p.save(f"{output}data_umap_representation_{appendix}.pdf")

    # p("joint", x="X", y="Y", hue="Data", kind="scatter", marginal_kws={'common_norm': False})
    # legend_position = (0.5, -0.2)
    # p.move_legend("upper center", obj=p.plot.ax_joint, bbox_to_anchor=legend_position, ncol=3)
    # p.save(f"{output}data_umap_representation_scatter_{appendix}.pdf")

    # no_diff_df = df[df["Data"] != labels[-1]]
    # only_diff_df = df[df["Data"] == labels[-1]]
    # p = plt(no_diff_df, format=["pdf", "png"])
    # p("joint", x="X", y="Y", hue="Data", kind="kde", fill=True, thresh=0.05, alpha=.8, marginal_kws={'common_norm': False})
    # sct_ax = sns.scatterplot(data=only_diff_df, x="X", y="Y", marker='x', c=["tab:green"], ax=p.plot.figure.axes[0])
    # handles, labels = p.plot.figure.axes[0].get_legend_handles_labels()
    # print(handles, labels)
    # p.move_legend("upper center", obj=p.plot.ax_joint, bbox_to_anchor=legend_position, ncol=3)
    # handles, labels = p.plot.figure.axes[0].get_legend_handles_labels()
    # print(handles, labels)
    # p.set_legend(x=legend_position[0],
    #              y=legend_position[1],
    #              loc="upper center",
    #              ncol=3,
    #              labels=["G", "B"])
    data = df["Data"].values
    data = np.where(data == "Good", 0, data)
    data = np.where(data == "Bad", 1, data)
    data = np.where(data == "Good2", 2, data)
    data = np.where(data == "Bad2", 3, data)
    df["Data"] = data
    df["Data"] = pd.to_numeric(df["Data"])

    colors_1 = ["#4daf4a", "#377eb8"]
    cm = sns.color_palette(colors_1)
    p = plt(df, format=["pdf", "png"], palette=cm)
    p("joint", x="X", y="Y", hue="Data", hue_order=[0, 1], kind="kde", fill=True, thresh=0.05, alpha=.8, marginal_kws={'common_norm': False})

    # colors = ["#377eb8", "#e41a1c"]
    colors= ["#4daf4a", "#377eb8"]
    cm = sns.color_palette(colors)
    p.plot.plot_joint(sns.scatterplot, markers=['x', 'x'], hue="Data", hue_order=[2, 3], palette=cm, linewidths=0.0)
    # p.plot.set_axis_labels()
    ax = mplt.gca()
    # ax.legend()
    # ug_patch = mpatches.Patch(color=colors_1[0], label=r"$\hat{U}_{Bad}$")
    # ub_patch = mpatches.Patch(color=colors_1[1], label=r"$\hat{U}_{Good}$")
    b_patch = mpatches.Patch(color=colors_1[0], label=r"$\hat{G}$")
    g_patch = mpatches.Patch(color=colors_1[1], label=r"$\hat{B}$")
    point_g = Line2D([0], [0], label=r"$\hat{U}_{Good}$", marker='o', markersize=10,
                   markeredgecolor=colors[0], markerfacecolor=colors[0], linestyle='')
    point_b = Line2D([0], [0], label=r"$\hat{U}_{Bad}$", marker='o', markersize=10,
                   markeredgecolor=colors[1], markerfacecolor=colors[1], linestyle='')
    ax.legend(handles=[g_patch, b_patch, point_g, point_b], ncol=1, loc="upper left")
    # ax.get_legend().remove()
    # p.move_legend("upper center", obj=p.plot, bbox_to_anchor=legend_position, ncol=3)
    p.save(f"{output}data_umap_representation_difficultScatter_{appendix}.pdf")

if __name__ == "__main__":
    main()
