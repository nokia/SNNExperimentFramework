# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import seaborn as sns
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib as mplt
# from embeddingTSNE import Transform
from SNN2.src.io.files import FileHandler as FH

from SNN2.src.plot.plotter import plotter as plt

from typing import Any, Callable, Dict, List, Optional

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="",
                    action="store", help="Define the appendix")
parser.add_argument("-p", "--phase", dest="phase", default=False,
                    action="store_true", help="Activate the phase flag")

def two_actions_next_state(state: List[float], probabilities: List[float], delta: float = 0.1):
    next_state = [state[0]+delta, state[0]-delta]
    return next_state, probabilities

def pivot(df, *args, **kwargs):
    default_kw = {
                "index": ["Episode"],
                "columns": ["Step"],
                "values": "Value",
                "dropna": True,
                "fill_value": 0.0,
                "aggfunc": np.mean
            }
    default_kw.update(kwargs)
    return df.pivot_table(*args, **default_kw)

def pivot_heatmap(df,
                  output,
                  pivot_kwargs: Optional[Dict[str, Any]] = None,
                  heatmap_kwargs: Optional[Dict[str, Any]] = None,
                  set_kwargs: Optional[Dict[str, Any]] = None) -> None:

    default_pivot_kw = {}
    if pivot_kwargs is not None:
        default_pivot_kw.update(pivot_kwargs)

    default_heatmap_ks = {
                "cmap": "tab20c",
                "xticklabels": 100
            }
    if heatmap_kwargs is not None:
        default_heatmap_ks.update(heatmap_kwargs)

    default_set = {
                "title": "Average Reward evolution Step X Evaluation",
                "ylabel": "Evaluation",
                "xlabel": "Step"
            }
    if set_kwargs is not None:
        default_set.update(set_kwargs)

    pvt_df = pivot(df, **default_pivot_kw)
    p = plt(pvt_df, format=["pdf", "png"])
    p("heatmap", **default_heatmap_ks)
    p.set(**default_set)
    p.save(output)


def plot_reward(df: pd.DataFrame,
                output: str,
                appendix: Optional[str] = "",
                **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "Reward"].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    # tmp_df["Step"] = tmp_df["Step"].apply(pd.to_numeric)
    # tmp_df["Episode"] = tmp_df["Episode"].astype("string")
    # avg_df = tmp_df.copy().groupby(["Episode", "Step"])["Value"].mean().reset_index()
    # avg_df = avg_df[avg_df["Step"] < 10]
    # order = avg_df["Episode"].value_counts().index
    # p = plt(avg_df, format=["pdf", "png"])
    # p("multiDimEvolution", "Step", "Value",
    #   facet_kwargs={
    #         "row": "Episode",
    #         "aspect": 10,
    #         "row_order": order,
    #         "despine": False
    #     }, map_fct=sns.lineplot)
    # p("kde", x="Value", hue="Episode")
    # p.set(ylim=(0.0, 2.0))
    # p("multiLineplot", "Episode", x="Step", y="Value")
    # p.save(output.replace('.pdf','_lineStack.pdf'))
    # return

    output_c = output.replace('.pdf', '_clrB.pdf')
    pivot_heatmap(tmp_df, output_c, heatmap_kwargs={"vmin": 0.0})
    pivot_heatmap(tmp_df, output_c.replace('.pdf','_STD.pdf'),
                  pivot_kwargs={"aggfunc": np.std},
                  set_kwargs={"title": "Average Reward STD Step X Evaluation"})
    pivot_heatmap(tmp_df, output_c.replace('.pdf','_VAR.pdf'),
                  pivot_kwargs={"aggfunc": np.var},
                  set_kwargs={"title": "Average Reward VAR Step X Evaluation"})
    pivot_heatmap(tmp_df, output, heatmap_kwargs={"vmin": 0.0, "cmap": "viridis"})
    pivot_heatmap(tmp_df, output.replace('.pdf','_STD.pdf'),
                  heatmap_kwargs={"cmap": "viridis_r"},
                  pivot_kwargs={"aggfunc": np.std, "fill_value": np.NaN},
                  set_kwargs={"title": "Average Reward STD Step X Evaluation"})
    pivot_heatmap(tmp_df, output.replace('.pdf','_VAR.pdf'),
                  heatmap_kwargs={"cmap": "viridis_r"},
                  pivot_kwargs={"aggfunc": np.var, "fill_value": np.NaN},
                  set_kwargs={"title": "Average Reward VAR Step X Evaluation"})

def state_to_ndarray(states: np.ndarray) -> np.ndarray:
    def filter_func(item):
        item = item.replace('[','').replace(']','')
        item = ",".join(item.split())
        m = tuple(list(map(float, item.split(','))))
        return m

    return np.stack(np.vectorize(filter_func)(states), axis=0)

def plot_margin(df: pd.DataFrame,
                output: str,
                appendix: Optional[str] = "",
                **kwargs) -> None:
    if FH.exists(output):
        return

    grouped_df = df.copy().groupby(["Experiment", "Episode", "Step"])["State"].first().reset_index()
    grouped_df["Value"] = state_to_ndarray(grouped_df["State"].values)[0, :]
    grouped_df.drop(["State"], axis=1, inplace=True)

    pivot_heatmap(grouped_df, output, pivot_kwargs={"fill_value": np.NaN},
                  set_kwargs={"title": "Average Margin evolution Step X Evaluation"})
    pivot_heatmap(grouped_df, output.replace('.pdf','_MEDIAN.pdf'),
                  pivot_kwargs={"fill_value": np.NaN, "aggfunc": np.median},
                  heatmap_kwargs={"norm": mplt.colors.LogNorm()},
                  set_kwargs={"title": "MEDIAN Margin evolution Step X Evaluation"})
    pivot_heatmap(grouped_df, output.replace('.pdf','_STD.pdf'),
                  pivot_kwargs={"fill_value": np.NaN, "aggfunc": np.std},
                  set_kwargs={"title": "STD Margin evolution Step X Evaluation"})
    pivot_heatmap(grouped_df, output.replace('.pdf','_VAR.pdf'),
                  pivot_kwargs={"fill_value": np.NaN, "aggfunc": np.var},
                  set_kwargs={"title": "VAR Margin evolution Step X Evaluation"})

    last_value = grouped_df.groupby(["Experiment", "Episode"])["Value"].last().reset_index()
    p = plt(last_value, format=["pdf", "png"])
    p("violin", x="Episode", y="Value", cut=0)
    p.set(title="Last step margin value",
          xlabel="Evaluation",
          ylabel="Margin")
    p.save(output.replace('.pdf','_last_margin_violin.pdf'))
    p("boxplot", x="Episode", y="Value")
    p.set(title="Last step margin value",
          xlabel="Evaluation",
          ylabel="Margin")
    p.save(output.replace('.pdf','_last_margin_boxplot.pdf'))

    # final_value = grouped_df.groupby(["Experiment"])["Value"].last().reset_index()
    # print(final_value["Value"].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9]))

def plot_accuracy(df: pd.DataFrame,
                  output: str,
                  appendix: Optional[str] = "",
                  **kwargs) -> None:
    if FH.exists(output):
        return
    tmp_df = df[df["Statistic"].isin(["TP", "FP", "TN", "FN", "U"])].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    sum_df = tmp_df.groupby(["Experiment", "Episode", "Step"])["Value"].sum().reset_index()
    total = sum_df["Value"]
    correct = df[df["Statistic"].isin(["TP", "TN"])].copy()
    correct["Value"] = correct["Value"].apply(pd.to_numeric)
    correct = correct.groupby(["Experiment", "Episode", "Step"])["Value"].sum().reset_index()
    correct = correct["Value"]
    accuracy = correct/total
    sum_df["Value"] = accuracy
    accuracy_lastStep = sum_df.groupby(["Experiment", "Episode"]).tail(1).reset_index()
    p = plt(accuracy_lastStep, format=["pdf", "png"])
    p("line", x="Episode", y="Value", **kwargs)
    p.set(ylabel="Accuracy [0-1]", xlabel="Evaluation", title="OTT Dval Accuracy per Evaluation")
    p.save(output)

def get_subset(df: pd.DataFrame,
               precision: List[str],
               column: str,
               value: str) -> pd.DataFrame:
    correct = df[df[column] == value].copy()
    index1 = pd.MultiIndex.from_arrays([df[col] for col in precision])
    index2 = pd.MultiIndex.from_arrays([correct[col] for col in precision])
    return df.loc[index1.isin(index2)].copy()

def get_action_df(df) -> pd.DataFrame:
    tmp_df = df[df["Statistic"] == "ActionProbs"].copy()
    probs = state_to_ndarray(tmp_df["Value"].values)
    probs = np.around(probs, 4)
    actions = {f"action-{i}": prob for i, prob in enumerate(probs)}
    act_df = pd.concat([
        pd.DataFrame({
                "Experiment": tmp_df["Experiment"].values,
                "Episode": tmp_df["Episode"].values,
                "Step": tmp_df["Step"].values,
                "Action": k,
                "Value": v
            }) for k, v in actions.items()
        ])
    return act_df

def plot_exploitation_probability(df: pd.DataFrame,
                                  output: str,
                                  appendix: Optional[str] = "",
                                  **kwargs) -> None:
    if FH.exists(output):
        return
    exploit_df = get_subset(df, ["Experiment", "Episode", "Step"], "Value", "Exploitation")
    act_df = get_action_df(exploit_df)

    p = plt(act_df, format=["pdf", "png"])
    p("violin", x="Episode", y="Value", hue="Action", split=True, cut=0, **kwargs)
    p.set(ylabel="Probability [0-1]", xlabel="Evaluation", title="Action prob. evolution in Exploitation per Evaluation")
    p.move_legend("lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    p.save(output)

def plot_exploitation_probability_umap(df: pd.DataFrame,
                                       output: str,
                                       appendix: Optional[str] = "",
                                       **kwargs) -> None:
    print("FIX THE MEMORY PROBLEM")
    return
    exploit_df = get_subset(df, ["Experiment", "Episode", "Step"], "Value", "Exploitation")
    # act_df = get_action_df(exploit_df)
    exploit_df = exploit_df[exploit_df["Episode"] == 9]
    act_probs = exploit_df[exploit_df["Statistic"] == "ActionProbs"]
    probs = state_to_ndarray(act_probs["Value"].values).T
    probs = np.around(probs, 4)
    action = exploit_df[exploit_df["Statistic"] == "Action"]["Value"].values
    state = state_to_ndarray(act_probs["State"].values).T
    dim = np.concatenate([state, probs], axis=1)
    umap_df = Transform.umap2df(dim)
    umap_df["Action"] = action
    p = plt(umap_df, format=["pdf", "png"])
    p("scatter", x="X", y="Y", hue="Action")
    p.save(output)

def plot_exploitation_probability_heatmap(df: pd.DataFrame,
                                          output: str,
                                          appendix: Optional[str] = "",
                                          **kwargs) -> None:
    if FH.exists(output):
        return
    exploit_df = get_subset(df, ["Experiment", "Episode", "Step"], "Value", "Exploitation")
    act_df = get_action_df(exploit_df)
    p = plt(act_df, format=["pdf", "png"])
    p("heatplot", col="Action",
      pivot_kwarg={'index': ["Episode"], "columns": ["Step"], "values": "Value", "dropna": True, "fill_value": np.NaN, "aggfunc": np.mean},
      heatmap_kwarg={'xticklabels': 200, 'cmap': "tab20c", "vmin": 0.0, "vmax": 1.0})
    p.save(output)

    p("heatplot", col="Action",
      pivot_kwarg={'index': ["Episode"], "columns": ["Step"], "values": "Value", "dropna": True, "fill_value": np.NaN, "aggfunc": np.std},
      heatmap_kwarg={'xticklabels': 200, 'cmap': "tab20c"},
      cbar_kws={'label': 'P STD'})
    p.save(output.replace('.pdf', '_STD.pdf'))

    p("heatplot", col="Action",
      pivot_kwarg={'index': ["Episode"], "columns": ["Step"], "values": "Value", "dropna": True, "fill_value": np.NaN, "aggfunc": np.var},
      heatmap_kwarg={'xticklabels': 200, 'cmap': "tab20c"},
      cbar_kws={'label': 'P VAR'})
    p.save(output.replace('.pdf', '_VAR.pdf'))

def all_rl_plot(df: pd.DataFrame,
                output: str,
                appendix: Optional[str] = "") -> None:
    ott_accuracy_file = f"{output}/OTT_accuracy{appendix}.pdf"
    plot_accuracy(df, ott_accuracy_file)
    exploitation_probabilities_file = f"{output}/OTT_exploitation_probabilities{appendix}.pdf"
    plot_exploitation_probability(df, exploitation_probabilities_file)
    exploitation_probabilities_heatmap_file = f"{output}/OTT_exploitation_probabilities_heatmap{appendix}.pdf"
    plot_exploitation_probability_heatmap(df, exploitation_probabilities_heatmap_file)
    reward_file = f"{output}/OTT_reward_evolution{appendix}.pdf"
    plot_reward(df, reward_file)
    margin_file = f"{output}/MNO_margin_evolution{appendix}.pdf"
    plot_margin(df, margin_file)
    # exploitation_probabilities_umap_file = f"{output}/OTT_exploitation_probabilities_umap{appendix}.pdf"
    # plot_exploitation_probability_umap(df, exploitation_probabilities_umap_file)

def termination_boxplot(df: pd.DataFrame,
                        output: str,
                        appendix: Optional[str] = "") -> None:
    tmp_df = df.copy().groupby(["Experiment"]).last().reset_index()
    plt_df = tmp_df[["Origin", "Episode"]]

    p = plt(plt_df, format=["pdf", "png"])
    p("boxplot", x='Origin', y='Episode')
    p.set(xlabel="Approach", ylabel="# Episodes to convergence", title="Number of episodes required to reach convergence")
    p.save(f"{output}/ConvergenceEpisodes{appendix}.pdf")

def plot_convergence_distribution(df: pd.DataFrame,
                                  output: str,
                                  appendix: Optional[str] = "") -> None:
    r_df = df[df["Phase"] == "RL_Training"].copy()
    r_df = r_df.copy().groupby(["Origin", "Experiment"]).tail(1).reset_index()
    kde_df = r_df[["Origin", "Episode"]].copy()

    p = plt(kde_df, format=["pdf", "png"])
    p("histplot", x='Episode', hue="Origin", kde=True, binwidth=1)
    p.set(xlabel="Episode", ylabel="# Converged experiments", title="Experiment convergence distribution")
    p.save(f"{output}/ConvergenceEpisodesDistribution{appendix}.pdf")

def main():
    options = parser.parse_args()
    input: FH = FH(options.csv_file, create=False)
    output: str = options.output_folder
    appendix: str = options.appendix
    phase: bool = options.phase

    df = pd.read_csv(input.path)

    if phase:
        train_df = df[df["Phase"] == "RL_Training"].copy()
        exploit_df = df[df["Phase"] == "RL_Exploitation"].copy()
        exploit_df["Episode"] = exploit_df["Episode"].values-10

        # termination_boxplot(train_df, output, appendix=f"{appendix}_RL-Train")
        plot_convergence_distribution(train_df, output, appendix=f"{appendix}_RL-Train")
        # all_rl_plot(train_df, output, appendix=f"{appendix}_RL-Train")
        # all_rl_plot(exploit_df, output, appendix=f"{appendix}_RL-Exploitation")


if __name__ == "__main__":
    main()
