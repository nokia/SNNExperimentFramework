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
parser.add_argument("-i", "--index", dest="index", default=0,
                    action="store", help="Define the index of the figure")

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    output: DH = DH(options.output_folder)
    index: int = options.index
    sns.set_theme(context="paper")
    # mplt.style.use(['science','ieee'])

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    cm = sns.color_palette(colors)
    font_file = '/home/mattia/src/SNNFrameWork/Fonts/helvetica.ttf'
    fm.fontManager.addfont(font_file)
    # font_manager.fontManager.ttflist.extend(font_list)
    custom_params = {'figure.figsize':(8,4), "font.family": 'helvetica', "pdf.fonttype": 42, "ps.useafm": True}
    sns.set_theme(font="helvetica", rc=custom_params)

    df = pd.read_csv(input)
    df = df[df["KPI"] == "Flag"]

    first_it = df[df["Iteration"] == 0].copy()
    print(f"first iteration:\nn-samples: {len(first_it)}")
    print(f"U->G: {len(first_it[first_it['Values'] == 0.0])}")
    print(f"U->B: {len(first_it[first_it['Values'] == 1.0])}")
    print(f"U->U: {len(first_it[first_it['Values'] == 2.0])}")
    other_it = df[df["Iteration"] > 0].copy()

    node_labels = [
                    "U",
                    "G0", "B0", "U0",
                    "G1", "B1", "U1",
                    "G2", "B2", "U2",
                    "G3", "B3", "U3",
                    "G4", "B4", "U4",
                    "G5", "B5", "U5",
                    "G6", "B6", "U6",
                    "G7", "B7", "U7",
                    "G8", "B8", "U8",
                  ]
    node_dict = {y:x for x, y in enumerate(node_labels)}

    def transition(source_it, dst_it):
        source_df = df[df["Iteration"] == source_it]
        dst_df = df[df["Iteration"] == dst_it]
        source_g_df = source_df[source_df["Values"] == 0.0]
        source_b_df = source_df[source_df["Values"] == 1.0]
        source_u_df = source_df[source_df["Values"] == 2.0]
        dst_g_df = dst_df[dst_df["Values"] == 0.0]
        dst_b_df = dst_df[dst_df["Values"] == 1.0]
        dst_u_df = dst_df[dst_df["Values"] == 2.0]

        source_g_ids = source_g_df["SampleID"].values
        source_b_ids = source_b_df["SampleID"].values
        source_u_ids = source_u_df["SampleID"].values
        dst_g_ids = dst_g_df["SampleID"].values
        dst_b_ids = dst_b_df["SampleID"].values
        dst_u_ids = dst_u_df["SampleID"].values

        src_g_dst_g = len(np.intersect1d(source_g_ids, dst_g_ids))
        src_g_dst_b = len(np.intersect1d(source_g_ids, dst_b_ids))
        src_g_dst_u = len(np.intersect1d(source_g_ids, dst_u_ids))
        src_b_dst_g = len(np.intersect1d(source_b_ids, dst_g_ids))
        src_b_dst_b = len(np.intersect1d(source_b_ids, dst_b_ids))
        src_b_dst_u = len(np.intersect1d(source_b_ids, dst_u_ids))
        src_u_dst_g = len(np.intersect1d(source_u_ids, dst_g_ids))
        src_u_dst_b = len(np.intersect1d(source_u_ids, dst_b_ids))
        src_u_dst_u = len(np.intersect1d(source_u_ids, dst_u_ids))

        print(src_g_dst_g, src_b_dst_g, src_u_dst_g)
        print(src_g_dst_b, src_b_dst_b, src_u_dst_b)
        print(src_g_dst_u, src_b_dst_u, src_u_dst_u)
        # sources = [
        #             f"G{source_it}", f"G{source_it}", f"G{source_it}",
        #             f"B{source_it}", f"B{source_it}", f"B{source_it}",
        #             f"U{source_it}", f"U{source_it}", f"U{source_it}",
        #           ]
        sources = [
                    f"G{source_it}", f"B{source_it}", f"U{source_it}",
                    f"G{source_it}", f"B{source_it}", f"U{source_it}",
                    f"G{source_it}", f"B{source_it}", f"U{source_it}",
                  ]
        destinations = [
                        f"G{dst_it}", f"G{dst_it}", f"G{dst_it}",
                        f"B{dst_it}", f"B{dst_it}", f"B{dst_it}",
                        f"U{dst_it}", f"U{dst_it}", f"U{dst_it}",
                       ]
        values = [
                    src_g_dst_g, src_b_dst_g, src_u_dst_g,
                    src_g_dst_b, src_b_dst_b, src_u_dst_b,
                    src_g_dst_u, src_b_dst_u, src_u_dst_u,
                 ]
        return sources, destinations, values

    sources = ["U", "U", "U"]
    destinations = ["G0", "B0", "U0"]
    values = [len(first_it[first_it['Values'] == 0.0]),
              len(first_it[first_it['Values'] == 1.0]),
              len(first_it[first_it['Values'] == 2.0])]

    for src, dst in zip(range(0,8),range(1,9)):
        s, d, v = transition(src, dst)
        sources += s
        destinations += d
        values += v

    source_node = [node_dict[x] for x in sources]
    target_node = [node_dict[x] for x in destinations]
    print(source_node)
    print(target_node)
    print(values)

    node_color = [
                    "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                    "#4daf4a", "#377eb8", "#e41a1c",
                 ]

    node_label_color = {x:y for x, y in zip(node_labels, node_color)}
    link_color = [node_label_color[x] for x in destinations]

    link_color = ['rgba({},{},{}, 0.4)'.format(
                    hex_to_rgb(x)[0],
                    hex_to_rgb(x)[1],
                    hex_to_rgb(x)[2]) for x in link_color]

    fig = go.Sankey( # The plot we are interest
            # This part is for the node information
            arrangement='snap',
            node = dict(
                color = node_color,
                x=[0,
                   0.1,0.1,0.1,
                   0.2,0.2,0.2,
                   0.3,0.3,0.3,
                   0.4,0.4,0.4,
                   0.5,0.5,0.5,
                   0.6,0.6,0.6,
                   0.7,0.7,0.7,
                   0.8,0.8,0.8,
                   0.9,0.9,0.9],
                y=[0.5,
                   0.1, 0.5, 0.85,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   0.1, 0.5, 0.9,
                   ]
            ),
            # This part is for the link information
            link = dict(
                source = source_node,
                target = target_node,
                value = values,
                color = link_color,
            ))

    legend = []
    legend_entries = [
        [node_color[1], r"$Good$"],
        [node_color[2], r"$Bad$"],
        [node_color[3], r"$\hat{U}$"],
    ]
    for entry in legend_entries:
        legend.append(
            go.Scatter(
                mode="markers",
                x=[None],
                y=[None],
                marker=dict(size=10, color=entry[0], symbol="square"),
                name=entry[1],
            )
        )

    traces = [fig] + legend
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 20, 'b': 0},
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig = go.Figure(data=traces, layout=layout)

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.06,
        xanchor="center",
        x=0.45,
        font=dict(
            size=34,
        ),
        bordercolor="Black",
        borderwidth=1,
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # With this save the plots
    fig.write_image(f"{output.path}/sankey-{index}.svg", width=1000, height=500)
    # plot(fig,
    #      image_filename=f"{output.path}/sankey.png",
    #      image='png',
    #      image_width=1000,
    #      image_height=600
    # )

if __name__ == "__main__":
    main()
