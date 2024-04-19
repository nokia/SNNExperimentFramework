# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

import seaborn as sns
import re
import argparse
import numpy as np
import pandas as pd
from SNN2.src.io.files import FileHandler as FH

from SNN2.src.plot.plotter import plotter as plt

from typing import Callable, Dict, List, Optional

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="",
                    action="store", help="Define the appendix")

def two_actions_next_state(state: List[float], probabilities: List[float], delta: float = 0.1):
    next_state = [state[0]+delta, state[0]-delta]
    return next_state, probabilities

def plot_single_evolution(df: pd.DataFrame,
                          output: str,
                          episode: int = 0,
                          appendix: Optional[str] = "") -> None:
    df = df[df["Episode"] == episode]
    reward_output = f"{output}/RL_reward_evolution_episode_{episode}{appendix}.pdf"
    plot_reward(df, reward_output, appendix=appendix, title=f"Reward evolution episode {episode}")
    accuracy_output = f"{output}/RL_accuracy_evolution_episode_{episode}{appendix}.pdf"
    plot_accuracy(df, accuracy_output, appendix=appendix, title=f"Accuracy evolution episode {episode}")
    accuracy_output = f"{output}/RL_margin_evolution_episode_{episode}{appendix}.pdf"
    plot_margin(df, accuracy_output, appendix=appendix, title=f"Margin evolution episode {episode}")
    # conf_matrix_output=f"{output}/RL_confusion_matrix_evolution_episode{episode}{appendix}.pdf"
    # plot_confusion_matrix(df, conf_matrix_output, appendix=appendix, title=f"Confusion matrix evolution episode {episode}")
    action_prob_output = f"{output}/RL_action_probabilities_evolution_episode{episode}{appendix}.pdf"
    plot_action_probability(df, action_prob_output, appendix=appendix, title=f"Action probabilities episode {episode}")
    # chosen_action_output =f"{output}/RL_action_chosen_evolution_episode{episode}{appendix}.pdf"
    # plot_action_chosen(df, chosen_action_output, appendix=appendix, title=f"Action chosen episode {episode}")
    # transition_matrix_output =f"{output}/RL_transition_matrix_episode{episode}{appendix}.csv"
    # plot_transition_matrix(df, transition_matrix_output, appendix=appendix, title=f"Transition matrix episode {episode}",
    #                        get_next_state=two_actions_next_state)

def plot_reward(df: pd.DataFrame,
                output: str,
                appendix: Optional[str] = "",
                title: Optional[str] = "title",
                average: bool = True,
                last_steps: Optional[int] = 10,
                **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "Reward"].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    last_steps = len(tmp_df["Value"].values) if last_steps is None else last_steps
    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)

    if average:
        last_n_rewards = tmp_df["Value"].values[-last_steps:]
        avg_rew = np.average(last_n_rewards)
        if avg_rew > 2.45:
            print(f"{title} - {avg_rew}")
        p.plot.axhline(y=avg_rew, linestyle='dashed', linewidth=1.5, color='red')

    p.set(ylabel="Reward", title=title)
    p.save(output)

def avg_reward(df: pd.DataFrame,
               output: str,
               appendix: Optional[str] = "",
               title: Optional[str] = "title",
               average: bool = True,
               last_step: int = 10,
               **kwargs) -> None:
    if FH.exists(output):
        return

    avg_r = []
    for ep in df["Episode"].unique():
        sub_df = df[(df["Episode"] == ep) & (df["Statistic"] == "Reward")].copy()
        sub_df["Value"] = sub_df["Value"].apply(pd.to_numeric)
        last_n_rewards = sub_df["Value"].values[-last_step:]
        avg_rew = np.average(last_n_rewards)
        avg_r.append(avg_rew)

    avg_df = pd.DataFrame({"Episode": range(len(avg_r)), "Value": avg_r})

    p = plt(avg_df, format=["pdf", "png"])
    p("line", x="Episode", y="Value", **kwargs)
    p.set(ylabel="Reward", title=title)
    p.save(output)

def plot_reward_heatmap(df: pd.DataFrame,
                        output: str,
                        appendix: Optional[str] = "",
                        title: Optional[str] = "title",
                        **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "Reward"].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    map_df = tmp_df.pivot_table(index=["Episode"],
                                columns=["Step"],
                                values="Value",
                                dropna=True,
                                fill_value=0.0)
    p = plt(map_df, format=["pdf", "png"])
    p("heatmap", vmin=0.0, vmax=2.0, xticklabels=100, cmap='tab20c', **kwargs)
    p.set(title=title,
          ylabel="SL-Cycle",
          xlabel="Step")
    p.save(output)

def state_to_ndarray(states: np.ndarray) -> np.ndarray:
    def filter_func(item):
        item = item.replace('[','').replace(']','')
        item = ",".join(item.split())
        m = tuple(list(map(float, item.split(','))))
        return m

    return np.stack(np.vectorize(filter_func)(states), axis=0)

def plot_margin_heatmap(df: pd.DataFrame,
                        output: str,
                        appendix: Optional[str] = "",
                        **kwargs) -> None:
    if FH.exists(output):
        return

    grouped_df = df.copy().groupby(["Episode", "Step"])["State"].first().reset_index()
    grouped_df["Value"] = state_to_ndarray(grouped_df["State"].values)[0, :]
    grouped_df.drop(["State"], axis=1, inplace=True)
    min_val = min(grouped_df["Value"].values)
    avg_df = grouped_df.groupby(["Episode", "Step"])["Value"].mean().reset_index()
    map_df = avg_df.pivot_table(index=["Episode"],
                                columns=["Step"],
                                values="Value",
                                dropna=True,
                                fill_value=min_val)
    p = plt(map_df, format=["pdf", "png"])
    p("heatmap", xticklabels=100, cmap='tab20c', **kwargs)
    p.set(title="Average Margin evolution Step X SL-Cycle",
          ylabel="SL-Cycle",
          xlabel="Step")
    p.save(output)

def plot_accuracy(df: pd.DataFrame,
                  output: str,
                  appendix: Optional[str] = "",
                  title: Optional[str] = "",
                  **kwargs) -> None:
    if FH.exists(output):
        return

    # tmp_df = df[df["Statistic"].isin(["TP", "FP", "TN", "FN", "undecided"])].copy()
    # tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    tmp_df = df[df["Statistic"] == "Accuracy"]
    tmp_df = tmp_df.drop(["State", "Statistic"], axis=1)
    tmp_df["Value"] = tmp_df["Value"].astype('float32')
    tmp_df = tmp_df.round({'Value': 4})
    # sum_df = tmp_df.groupby(["Episode", "Step"])["Value"].sum().reset_index()
    # total = sum_df["Value"]
    # correct = df[df["Statistic"].isin(["TP", "TN"])].copy()
    # correct["Value"] = correct["Value"].apply(pd.to_numeric)
    # correct = correct.groupby(["Episode", "Step"])["Value"].sum().reset_index()["Value"]
    # accuracy = correct/total
    # sum_df["Value"] = accuracy
    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)
    # for i in range(0, tmp_df["Step"].max(), 20):
    #     p.plot.axvline(x=i, linestyle='--', linewidth=0.1, color='black')
    p.set(ylabel="SNN Accuracy", title=title)
    p.save(output)

def plot_margin(df: pd.DataFrame,
                output: str,
                appendix: Optional[str] = "",
                title: Optional[str] = "",
                **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "Accuracy"].copy()
    margins = state_to_ndarray(tmp_df["State"].values)[0, :]
    tmp_df = pd.DataFrame({"Step": range(len(margins)),"Value": margins})
    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)
    for i in range(0, tmp_df["Step"].max(), 20):
        p.plot.axvline(x=i, linestyle='--', linewidth=0.1, color='black')
    p.set(ylabel="SNN Margin value", title=title)
    p.save(output)

def get_margin_chainging(m, p):
   tmp_p = np.round(p, 2)
   res = np.where(tmp_p == 0.50)[0]
   if len(res) == 0:
       return None
   return m[res[0]]

def plot_correlation(df: pd.DataFrame,
                     output: str,
                     appendix: Optional[str] = "",
                     title: Optional[str] = "",
                     **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "ActionProbs"].copy()
    margins = state_to_ndarray(tmp_df["State"].values)[0, :]
    probabilities = state_to_ndarray(tmp_df["Value"].values)
    print(margins.shape)
    print(probabilities.shape)
    p0 = probabilities[0, :]
    p1 = probabilities[1, :]
    print(p0.shape, p1.shape)
    limits = []
    for ep in tmp_df["Episode"].unique():
        max = tmp_df[tmp_df["Episode"] == ep]["Step"].max()
        limits.append(max)

    total = 0
    for i, l in enumerate(limits):
        p0_corr = np.corrcoef(margins[total:l+total], p0[total:l+total])[0, 1]
        p1_corr = np.corrcoef(margins[total:l+total], p1[total:l+total])[0, 1]
        margin_chainging = get_margin_chainging(margins[total:l+total], p0[total:l+total])
        total += l
        print(i, p0_corr, p1_corr, margin_chainging)
    # tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    # sum_df = tmp_df.groupby(["Episode", "Step"])["Value"].sum().reset_index()
    # total = sum_df["Value"]
    # correct = df[df["Statistic"].isin(["TP", "TN"])].copy()
    # correct["Value"] = correct["Value"].apply(pd.to_numeric)
    # correct = correct.groupby(["Episode", "Step"])["Value"].sum().reset_index()["Value"]
    # accuracy = correct/total
    # sum_df["Value"] = accuracy
    # p = plt(sum_df, format=["pdf", "png"])
    # p("line", x="Step", y="Value", **kwargs)
    # p.set(ylabel="OTT Accuracy", title=title)
    # p.save(output)

def plot_confusion_matrix(df: pd.DataFrame,
                          output: str,
                          appendix: Optional[str] = "",
                          title: Optional[str] = "",
                          **kwargs) -> None:
    if FH.exists(output):
        return
    tmp_df = df[df["Statistic"].isin(["TP", "FP", "TN", "FN", "U"])].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    p = plt(tmp_df, format=["pdf", "png"])
    p("lineStack", x="Step", y="Value", hue="Statistic", normalize=True, **kwargs)
    p.set(ylabel="# Samples", title=title)
    p.move_legend("lower center", bbox_to_anchor=(0.5, -0.32), ncol=5)
    p.save(output)

def plot_action_probability(df: pd.DataFrame,
                            output: str,
                            appendix: Optional[str] = "",
                            title: Optional[str] = "",
                            **kwargs) -> None:
    if FH.exists(output):
        return
    tmp_df = df[df["Statistic"] == "ActionProbs"].copy()
    new_df = pd.DataFrame(columns=["Episode", "Step", "Action", "Value"])
    for val in tmp_df.values:
        ep = val[0]
        step = val[1]
        act_probs = re.sub(' +', ' ', val[-1])
        act_probs = re.sub(' ]', ']', act_probs)
        act_probs = act_probs.replace('[', '').replace(']', '').replace(',', '').split(' ')
        act_probs = np.array(act_probs)
        act_probs = act_probs.astype(np.float)
        act_probs = np.around(act_probs, 4)
        probs = { f"action-{i}": prob for i, prob in enumerate(act_probs)}
        dff = pd.DataFrame({
                        "Episode": ep,
                        "Step": step,
                        "Action": probs.keys(),
                        "Value": probs.values()
                    })
        new_df = pd.concat([new_df, dff])
    p = plt(new_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", hue="Action", **kwargs)
    p.set(ylabel="Probability [0-1]", title=title)
    p.move_legend("lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    p.save(output)

def plot_action_chosen(df: pd.DataFrame,
                       output: str,
                       appendix: Optional[str] = "",
                       title: Optional[str] = "",
                       **kwargs) -> None:
    if FH.exists(output):
        return
    tmp_df = df[df["Statistic"] == "Action"].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)
    p.set(ylabel="# Action chosen", title=title, yticks=tmp_df["Value"].unique())
    p.save(output)


def plot_transition_matrix(df: pd.DataFrame,
                           output: str,
                           appendix: Optional[str] = "",
                           title: Optional[str] = "",
                           get_next_state: Optional[Callable] = None,
                           **kwargs) -> None:
    if FH.exists(output):
        return

    if get_next_state is None:
        raise Exception("A get_next_state function must be defined")

    tmp_df = df[df["Statistic"] == "ActionProbs"].copy()
    new_df = pd.DataFrame(columns=["State", "NextState", "Value"])
    for val in tmp_df.values:
        def str2list(obj: str, round: int = 4):
            res = re.sub(' +', ' ', obj)
            res = re.sub(' ]', ']', res)
            res = res.replace('[', '').replace(']', '').replace(',', '').split(' ')
            res = np.array(res)
            res = res.astype(np.float)
            res = np.around(res, round)
            return res

        state = str2list(val[2], round=1)
        act_probs = str2list(val[-1])
        nextState, probabilities = get_next_state(state, act_probs)
        dff = pd.DataFrame({
                        "State": state[0],
                        "NextState": nextState,
                        "Value": probabilities
                    })
        new_df = pd.concat([new_df, dff])
    new_df = new_df.drop_duplicates()
    new_df["NextState"] = new_df["NextState"].apply(pd.to_numeric)
    new_df["NextState"] = new_df["NextState"].round(decimals=2)
    new_df = new_df.pivot_table(index=["State"],
                                columns=["NextState"],
                                values="Value",
                                dropna=True,
                                fill_value=0.0)
    # p = plt(new_df, format=["pdf", "png"])
    # p("heatmap", **kwargs)
    # p.save(output)
    new_df.to_csv(output, index=True)

def get_action_df(df) -> pd.DataFrame:
    tmp_df = df[df["Statistic"] == "ActionProbs"].copy()
    probs = state_to_ndarray(tmp_df["Value"].values)
    probs = np.around(probs, 4)
    actions = {f"action-{i}": prob for i, prob in enumerate(probs)}
    act_df = pd.concat([
        pd.DataFrame({
                "Episode": tmp_df["Episode"].values,
                "Step": tmp_df["Step"].values,
                "Action": k,
                "Value": v
            }) for k, v in actions.items()
        ])
    return act_df

def get_subset(df: pd.DataFrame,
               precision: List[str],
               column: str,
               value: str) -> pd.DataFrame:
    correct = df[df[column] == value].copy()
    index1 = pd.MultiIndex.from_arrays([df[col] for col in precision])
    index2 = pd.MultiIndex.from_arrays([correct[col] for col in precision])
    return df.loc[index1.isin(index2)].copy()

def plot_probability_heatmap(df: pd.DataFrame,
                             output: str,
                             appendix: Optional[str] = "",
                             **kwargs) -> None:
    if FH.exists(output):
        return
    exploit_df = get_subset(df, ["Episode", "Step"], "Value", "Exploitation")
    act_df = get_action_df(exploit_df)
    p = plt(act_df, format=["pdf", "png"])
    p("heatplot", col="Action",
      pivot_kwarg={'index': ["Episode"], "columns": ["Step"], "values": "Value", "dropna": True, "fill_value": np.NaN},
      heatmap_kwarg={'xticklabels': 200, 'vmin': 0.0, 'vmax': 1.0, 'cmap': "tab20c"})
    p.save(output)

def main():
    options = parser.parse_args()
    input: FH = FH(options.csv_file, create=False)
    output: str = options.output_folder
    appendix: str = options.appendix

    df = pd.read_csv(input.path)

    for ep in df["Episode"].unique():
        plot_single_evolution(df, output, episode=ep, appendix=appendix)

    reward_output=f"{output}/RL_averageReward_evolution_{appendix}.pdf"
    last_stp = 10
    avg_reward(df, reward_output, appendix=appendix, last_step=last_stp, title=f"Avg Reward evolution, last {last_stp}")
    # reward_output=f"{output}/RL_reward_evolution_all_episodes{appendix}.pdf"
    # plot_reward(df, reward_output, appendix=appendix, title="Reward evolution on multiple episodes")
    # reward_output=f"{output}/RL_reward_heatmap_evolution_all_episodes{appendix}.pdf"
    # plot_reward_heatmap(df, reward_output, appendix=appendix, title="Reward evolution on multiple episodes")
    # accuracy_output=f"{output}/RL_accuracy_evolution_all_episodes{appendix}.pdf"
    # plot_accuracy(df, accuracy_output, appendix=appendix, title="Accuracy evolution on multiple episodes")
    # correlation_output=f"{output}/RL_correlation_evolution_all_episodes_separated{appendix}.pdf"
    # plot_correlation(df, correlation_output, appendix=appendix, title="Correlation evolution")
    # accuracy_output=f"{output}/RL_accuracy_evolution_all_episodes_separated{appendix}.pdf"
    # plot_accuracy(df, accuracy_output, appendix=appendix, title="Accuracy evolution on multiple episodes", hue="Episode")
    # margin_output=f"{output}/RL_margin_heatmap_evolution_all_episodes{appendix}.pdf"
    # plot_margin_heatmap(df, margin_output, appendix=appendix)
    # probability_output=f"{output}/RL_probability_heatmap_evolution_all_episodes{appendix}.pdf"
    # plot_probability_heatmap(df, probability_output, appendix=appendix)

if __name__ == "__main__":
    main()
