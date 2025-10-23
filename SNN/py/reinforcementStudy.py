# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import seaborn as sns
import re
import argparse
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.ndimage import shift
from SNN2.src.io.files import FileHandler as FH

from SNN2.src.plot.plotter import plotter as plt

from typing import Callable, Dict, List, Optional, Tuple

parser = argparse.ArgumentParser(usage="usage: grayPredEvaluation.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-l", "--loss", dest="loss_file", default="result/reinforcement_evolution_test.csv",
                    action="store", help="define the loss input file")
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
    # accuracy_output = f"{output}/RL_accuracy_evolution_episode_{episode}{appendix}.pdf"
    # multi_k_plot_accuracy(df, accuracy_output, appendix=appendix, title=f"Accuracy evolution episode {episode}")
    # accuracy_output = f"{output}/RL_mt_evolution_episode_{episode}{appendix}.pdf"
    # multi_k_plot_mt(df, accuracy_output, appendix=appendix, title=f"M_t evolution episode {episode}")
    # accuracy_output = f"{output}/RL_accuracy_evolution_episode_{episode}{appendix}.pdf"
    # plot_accuracy(df, accuracy_output, appendix=appendix, title=f"Accuracy evolution episode {episode}")
    # accuracy_output = f"{output}/RL_Weighted_accuracy_evolution_episode_{episode}{appendix}.pdf"
    # plot_weight_accuracy(df, accuracy_output, appendix=appendix, title=f"Accuracy evolution episode {episode}")
    accuracy_output = f"{output}/RL_margin_evolution_episode_{episode}{appendix}.pdf"
    # plot_margin(df, accuracy_output, appendix=appendix, title=r"$\eta_t$ evolution at convergence")
    plot_margin(df, accuracy_output, appendix=appendix)
    # conf_matrix_output=f"{output}/RL_confusion_matrix_evolution_episode{episode}{appendix}.pdf"
    # plot_confusion_matrix(df, conf_matrix_output, appendix=appendix, title=f"Confusion matrix evolution episode {episode}")
    action_prob_output = f"{output}/RL_action_probabilities_evolution_episode{episode}{appendix}.pdf"
    # plot_action_probability(df, action_prob_output, appendix=appendix, title=f"Act. p. evolution at convergence")
    plot_action_probability(df, action_prob_output, appendix=appendix)
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
                last_steps: Optional[int] = 100,
                **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "Reward"].copy()
    tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
    last_steps = len(tmp_df["Value"].values) if last_steps is None else last_steps
    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)

    # def moving_average(a, n=10) :
    #     ret = np.cumsum(a, dtype=float)
    #     ret[n:] = ret[n:] - ret[:-n]
    #     return ret[n - 1:] / n

    # wdw = 10
    # x = moving_average(tmp_df["Value"].values, n=wdw)
    # tmp_df = pd.DataFrame({'Step': range(len(x)), "moving_avg": x})
    # ax2 = p.plot.twinx()
    # sns.lineplot(data=tmp_df, x="Step", y="moving_avg", color="green", linestyle='dashed', markers=True, ax=ax2)
    # # ax2.set_ylim((-0.5, 0.5))
    # ax2.set_ylabel(f'Moving avg(r_t, n={wdw})')
    # # if average:
    # #     last_n_rewards = tmp_df["Value"].values[-last_steps:]
    # #     avg_rew = np.average(last_n_rewards)
    # #     if avg_rew >= 0.55:
    # #         print(f"{title} - {avg_rew}")
    # # print(np.trapz(x))
    # ax2.axhline(y=np.trapz(x), linewidth=1.5, color='red')
    # ax2.axhline(y=0.0, linewidth=0.5, color='black')
    # ax2.set_ylim((-0.5, 0.5))

    p.set(ylabel="Reward", ylim=(-1.1, 1.1), title=title)
    p.save(output)

def find_best_avg_subset(df: pd.DataFrame, n=10) -> Tuple[Tuple[int, int], float]:
    values = df["Value"].values
    sliding_avg = np.lib.stride_tricks.sliding_window_view(values, n).mean(axis=-1)
    i = np.argmax(sliding_avg)
    print(f"index: {i} with avg: {max(sliding_avg)}, values: {values[i:i+n]}")
    return (i, i+n), round(max(sliding_avg), 4)


def avg_reward(df: pd.DataFrame,
               output: str,
               appendix: Optional[str] = "",
               title: Optional[str] = "title",
               average: bool = True,
               last_step: int = -1,
               threshold: Optional[float] = None,
               gray_limit: int = 10,
               **kwargs) -> None:

    avg_r = []
    ep_list = df["Episode"].unique()[:-gray_limit] if gray_limit > 0 else df["Episode"].unique()
    for ep in ep_list:
        sub_df = df[(df["Episode"] == ep) & (df["Statistic"] == "Reward")].copy()
        sub_df["Value"] = sub_df["Value"].apply(pd.to_numeric)
        last_n_rewards = sub_df["Value"].values[-last_step:] if last_step > 0 else sub_df["Value"].values
        avg_rew = np.average(last_n_rewards)
        # if ep == 269:
        #     print(last_n_rewards)
        #     print(avg_rew)
        #     raise Exception
        avg_r.append(avg_rew)

    avg_df = pd.DataFrame({"Episode": range(len(avg_r)), "Value": avg_r})

    # print(avg_df["Value"])
    # best_subset, avg_value = find_best_avg_subset(avg_df, n=5)


    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    cm = sns.color_palette(colors)
    rcParams['figure.figsize'] = 8,4
    # rcParams['font.size'] = 80
    # sns.set(font_scale=2.5)
    # sns.set_style("white")

    p = plt(avg_df, format=["pdf", "png"])
    p.sns_set(font_scale=2.4)
    p.sns_set_api(sns.set_style, "white")
    p("line", x="Episode", y="Value", linewidth=2.0, **kwargs)
    # p.plot.axhline(y=avg_value, linestyle='dashed', linewidth=1.5, color='red')
    # p.plot.axvline(x=best_subset[0], linewidth=1, color='black')
    # p.plot.axvline(x=best_subset[1], linewidth=1, color='black')
    # p.plot.text(best_subset[0]-5, 1.3,f"{best_subset[0]}", fontsize=9)
    # p.plot.text(0, avg_value+0.01, f"{avg_value}", fontsize=9)
    # if threshold is not None:
    #     p.plot.axhline(y=threshold, linestyle='dashed', linewidth=1.5, color='red')
    p.set(ylabel="Avg. Reward", title=title)
    p.save(output)

def discount_rewards(rewards, gamma):
    t_steps = np.arange(rewards.size)
    r = rewards * gamma**t_steps
    r = r[::-1].cumsum()[::-1] / gamma**t_steps
    return r

def sum_reward(df: pd.DataFrame,
               output: str,
               appendix: Optional[str] = "",
               title: Optional[str] = "title",
               average: bool = True,
               last_step: int = -1,
               threshold: float = 0.2,
               gray_limit: int = 10,
               **kwargs) -> None:

    avg_r = []
    ep_list = df["Episode"].unique()[:-gray_limit] if gray_limit > 0 else df["Episode"].unique()
    for ep in ep_list:
        sub_df = df[(df["Episode"] == ep) & (df["Statistic"] == "Reward")].copy()
        sub_df["Value"] = sub_df["Value"].apply(pd.to_numeric)
        last_n_rewards = sub_df["Value"].values[-last_step:] if last_step > 0 else sub_df["Value"].values
        avg_rew = sum(last_n_rewards)
        avg_r.append(avg_rew)

    avg_df = pd.DataFrame({"Episode": range(len(avg_r)), "Value": avg_r})

    # best_subset, avg_value = find_best_avg_subset(avg_df)
    # raise Exception

    p = plt(avg_df, format=["pdf", "png"])
    p("line", x="Episode", y="Value", **kwargs)
    # p.plot.axhline(y=threshold, linestyle='dashed', linewidth=1.5, color='red')
    p.set(ylabel="Avg Reward", title=title)
    p.save(output)

def reward_uca(df: pd.DataFrame,
               output: str,
               appendix: Optional[str] = "",
               title: Optional[str] = "title",
               average: bool = True,
               last_step: int = -1,
               threshold: float = 0.2,
               gray_limit: int = 10,
               **kwargs) -> None:

    avg_r = []
    ep_list = df["Episode"].unique()[:-gray_limit] if gray_limit > 0 else df["Episode"].unique()
    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    wdw = 10
    for ep in ep_list:
        sub_df = df[(df["Episode"] == ep) & (df["Statistic"] == "Reward")].copy()
        sub_df["Value"] = sub_df["Value"].apply(pd.to_numeric)
        last_n_rewards = sub_df["Value"].values[-last_step:] if last_step > 0 else sub_df["Value"].values
        x = moving_average(last_n_rewards, n=wdw)
        area = np.trapz(x)
        avg_r.append(area)

    avg_df = pd.DataFrame({"Episode": range(len(avg_r)), "Value": avg_r})

    # best_subset, avg_value = find_best_avg_subset(avg_df)
    # raise Exception

    p = plt(avg_df, format=["pdf", "png"])
    p("line", x="Episode", y="Value", **kwargs)
    # p.plot.axhline(y=threshold, linestyle='dashed', linewidth=1.5, color='red')
    p.set(ylabel=f"Integral(Moving_avg(r_t, n={wdw}))", title=title)
    p.save(output)

def loss_plot(df: pd.DataFrame,
              output: str,
              appendix: Optional[str] = "",
              title: Optional[str] = "title",
              **kwargs) -> None:
    tmp_df = df.copy()
    tmp_df["loss"] = tmp_df["loss"].apply(pd.to_numeric)
    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="epoch", y="loss", **kwargs)
    p.set(ylabel="Loss value", xlabel="Epoch", title=title)
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
        # print("Filter funct")
        # print(item)
        item = item.replace('[','').replace(']','')
        # print(item)
        if "," not in item:
            item = ",".join(item.split())
        # print(item)
        m = tuple(list(map(float, item.split(','))))
        # print(m)
        # print("End filter funct")
        return m

    # elem = None
    # for st in states:
    #     if elem is None:
    #         elem = np.array([filter_func(st)])
    #     else:
    #         elem = np.vstack([elem, filter_func(st)])
    # print(elem)
    # elem_np = np.array([*elem])
    # print(elem_np)
    # raise Exception
    # all = np.stack(np.array(elem))
    # print(all)
    # raise Exception
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

def multi_k_plot_accuracy(df: pd.DataFrame,
                          output: str,
                          appendix: Optional[str] = "",
                          title: Optional[str] = "",
                          **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"].isin(["WeightedAccuracy", "normalAccuracy"])].copy()
    # tmp_df = df[df["Statistic"].isin(["WeightedAccuracy"])].copy()
    accuracies = state_to_ndarray(tmp_df["Value"].values)
    mean_acc = np.mean(accuracies, axis=0)
    tmp_df["Value"] = mean_acc

    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", hue="Statistic", **kwargs)
    # p.set(ylim=(0.88, None), ylabel="RL Accuracy (W_t)", title=title)
    p.set(ylabel="RL Accuracy (W_t)", title=title)

    # mt_df = df[df["Statistic"].isin(["RewardComAccuracy"])].copy()
    # mt_df["Value"] = mt_df["Value"].apply(pd.to_numeric)
    # ax2 = p.plot.twinx()
    # sns.lineplot(data=mt_df, x="Step", y="Value", color="crimson", dashes=True, ax=ax2)
    # ax2.set_ylabel('M_t')
    # ax2.set_ylim((1.8, None))

    p.move_legend("lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    p.save(output)


# def discounted_sum(gamma: float, x: tf.Tensor) -> tf.Tensor:
#     @tf.function
#     def aggregate(agg, x):
#         return gamma*agg + x
#     return tf.scan(aggregate, x)

# def multi_k_plot_mt(df: pd.DataFrame,
#                     output: str,
#                     appendix: Optional[str] = "",
#                     title: Optional[str] = "",
#                     **kwargs) -> None:
#     if FH.exists(output):
#         return

#     # tmp_df = df[df["Statistic"].isin(["RewardComAccuracy"])].copy()
#     # tmp_df["Value"] = tmp_df["Value"].apply(pd.to_numeric)
#     tmp_df = df[df["Statistic"].isin(["WeightedAccuracy"])].copy()
#     accuracies = state_to_ndarray(tmp_df["Value"].values).T
#     steps = tmp_df["Step"].values
#     new_df = pd.DataFrame()
#     for gamma in [0.5]:
#         acum_acc = []
#         for acc in accuracies:
#             accumulated = discounted_sum(gamma, tf.convert_to_tensor(acc)).numpy()[-1]
#             acum_acc.append(accumulated)
#         new_df = pd.concat([new_df, pd.DataFrame({"Step": steps, "Gamma": gamma, "Value": acum_acc})])

#     p = plt(new_df, format=["pdf", "png"])
#     p("line", x="Step", y="Value", hue="Gamma", **kwargs)
#     p.set(ylim=(1.86,1.87), ylabel="Acc. Comulative sum (M_t)", title=title)
#     p.move_legend("lower center", bbox_to_anchor=(0.5, -0.42), ncol=4)
#     p.save(output)

def plot_accuracy(df: pd.DataFrame,
                  output: str,
                  appendix: Optional[str] = "",
                  title: Optional[str] = "",
                  **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "Accuracy"]
    tmp_df = tmp_df.drop(["State", "Statistic"], axis=1)
    tmp_df["Value"] = tmp_df["Value"].astype('float32')
    tmp_df = tmp_df.round({'Value': 4})

    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)

    p.set(ylabel="SNN Accuracy", title=title)
    p.save(output)

def to_long(df, **kwargs):
    tmp_df = pd.wide_to_long(df, **kwargs)
    tmp_df.reset_index(inplace=True)
    return tmp_df

def plot_weight_accuracy(df: pd.DataFrame,
                         output: str,
                         appendix: Optional[str] = "",
                         title: Optional[str] = "",
                         limit: int = 10,
                         **kwargs) -> None:
    if FH.exists(output):
        return

    tmp_df = df[df["Statistic"] == "RewardAccuracy"]
    tmp_df = tmp_df.drop(["State", "Statistic"], axis=1)
    tmp_df["Value0"] = tmp_df["Value"].astype('float32')
    tmp_df = tmp_df.round({'Value0': 4})

    acc = np.array(tmp_df["Value0"].values)
    for i in range(1, limit):
        tmp_df[f"Value{i}"] = shift(acc, i, cval=np.NaN)
    tmp_df = tmp_df.drop(["Value"], axis=1)
    tmp_df = to_long(tmp_df, stubnames='Value', i=['Episode', 'Step'], j="Cycle", suffix=".*")
    tmp_df = tmp_df.dropna()

    # tmp_df = tmp_df[tmp_df["Step"].values % 10 == 0]

    p = plt(tmp_df, format=["pdf", "png"])
    p("line", x="Step", y="Value", **kwargs)

    p.set(ylabel=f"SNN Weight accuracy looking {limit} in the past", title=title)
    p.save(output)

def plot_margin(df: pd.DataFrame,
                output: str,
                appendix: Optional[str] = "",
                title: Optional[str] = "",
                **kwargs) -> None:
    if FH.exists(output):
        return

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    cm = sns.color_palette(colors)
    rcParams['figure.figsize'] = 8,4
    # rcParams['font.size'] = 300

    tmp_df = df[df["Statistic"] == "Action"].copy()
    margins = state_to_ndarray(tmp_df["State"].values)[0, :]
    tmp_df = pd.DataFrame({"Step": range(len(margins)),"Value": margins})
    tmp_df["Value"] *= 1200
    p = plt(tmp_df, format=["pdf"], palette=cm)
    p.sns_set(font_scale=2.4)
    p.sns_set_api(sns.set_style, "white")
    p("line", x="Step", y="Value", linewidth=2.0, **kwargs)
    # for i in range(0, tmp_df["Step"].max(), 20):
    #     p.plot.axvline(x=i, linestyle='--', linewidth=0.1, color='black')
    p.set(ylabel=r"$\eta_t$ value", title=title)
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

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    cm = sns.color_palette(colors)
    rcParams['figure.figsize'] = 8,4
    # rcParams['font.size'] = 80
    # sns.set(font_scale=2.5)
    # sns.set_style("white")

    tmp_df = df[df["Statistic"] == "ActionProbs"].copy()
    new_df = pd.DataFrame(columns=["Episode", "Step", "Action", "Value"])
    for val in tmp_df.values:
        ep = val[0]
        step = val[1]
        act_probs = re.sub(' +', ' ', val[-1])
        act_probs = re.sub(' ]', ']', act_probs)
        act_probs = act_probs.replace('[', '').replace(']', '').replace(',', '').split(' ')
        act_probs = np.array(act_probs)
        act_probs = act_probs.astype(float)
        act_probs = np.around(act_probs, 4)
        probs = { rf"$a_{i}$".replace("0", "+").replace("1", "-").replace("2", "="): prob for i, prob in enumerate(act_probs)}
        dff = pd.DataFrame({
                        "Episode": ep,
                        "Step": step,
                        "Action": probs.keys(),
                        "Value": probs.values()
                    })
        new_df = pd.concat([new_df, dff])
    p = plt(new_df, format=["pdf", "png"], palette=cm)
    p.sns_set(font_scale=2.4)
    p.sns_set_api(sns.set_style, "white")
    p("line", x="Step", y="Value", hue="Action", style="Action", linewidth=2.0, **kwargs)
    p.set(ylabel="Probability [0-1]", ylim=(0.1, 0.7), title=title)
    p.move_legend("upper center", ncol=3, fontsize='20', title_fontsize='25')
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
            res = res.astype(float)
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
    loss: FH = FH(options.loss_file, create=False)
    output: str = options.output_folder
    appendix: str = options.appendix

    df = pd.read_csv(input.path)
    loss_df = pd.read_csv(loss.path)

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    cm = sns.color_palette(colors)
    rcParams['figure.figsize'] = 8,4

    # for ep in df["Episode"].unique():
    #     prt = False
    #     if len(df["Episode"].unique()) < 200:
    #         prt = True
    #     elif ep % 5 == 0:
    #         prt = True

    #     if prt:
    #         plot_single_evolution(df, output, episode=ep, appendix=appendix)
    plot_single_evolution(df, output, episode=69, appendix=appendix)

    reward_output=f"{output}/RL_averageReward_evolution_{appendix}.pdf"
    last_stp = -1
    gray_limit = 10
    threshold = None
    # avg_reward(df, reward_output, threshold=threshold, gray_limit=gray_limit, appendix=appendix, last_step=last_stp, title=f"RL-training, Avg Reward evolution")
    avg_reward(df, reward_output, threshold=threshold, gray_limit=gray_limit, appendix=appendix, last_step=last_stp, title="")
    # reward_output=f"{output}/RL_sumReward_evolution_{appendix}.pdf"
    # sum_reward(df, reward_output, threshold=threshold, gray_limit=gray_limit, appendix=appendix, last_step=last_stp, title=f"Sum(r_t) for each episode")
    # reward_output=f"{output}/RL_AreaUnderCurve_evolution_{appendix}.pdf"
    # reward_uca(df, reward_output, threshold=threshold, gray_limit=gray_limit, appendix=appendix, last_step=last_stp, title=f"Sum(r_t) for each episode")
    reward_output=f"{output}/RL_loss_evolution_{appendix}.pdf"
    loss_plot(loss_df, reward_output, appendix=appendix, title=f"RL Loss value")
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
