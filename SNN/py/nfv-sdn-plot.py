from re import I
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as mplt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
import matplotlib.ticker as mtkr
from itertools import product
import subprocess
from matplotlib import rcParams

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
parser.add_argument("-e", "--eval_file", dest="evaluation_file", default="result/evaluation.csv",
                    action="store", help="define the evaluation csv input file")

# def raincloud(df, ticks, output_file,
#               title: str = "Test-Dataset Accuracy",
#               ylabel: str = "Accuracy [%]",
#               xlabel: str = "Loss Function",
#               **kwargs):
#     raincloud_kw = {
#                 "data": df,
#                 "x": "function",
#                 "y": "Value",
#                 "palette": "Set2",
#                 "bw": 0.2,
#                 "width_viol": .5,
#                 "alpha": .5,
#                 "dodge": True,
#                 "move": 0.2,
#                 "pointplot": False,
#                 "box_medianprops": {"linewidth": 1., "zorder": 10}
#             }
#     raincloud_kw.update(kwargs)

#     ax = pt.RainCloud(**raincloud_kw)
#     ax.set_xticklabels(ticks)
#     # mplt.title(title)
#     mplt.ylabel(ylabel)
#     mplt.xlabel(xlabel)
#     # handles, labels = ax.get_legend_handles_labels()
#     # ax.get_legend().remove()
#     # mplt.legend(handles[:len(df["Statistic"].unique())],
#     #             labels[:len(df["Statistic"].unique())],
#     #             bbox_to_anchor=(0.5, -0.40),
#     #             loc="lower center", ncol=2, borderaxespad=0,
#     #             title="Accuracy")
#     mplt.savefig(f"{output_file}.pdf", bbox_inches='tight')
#     mplt.savefig(f"{output_file}.png", bbox_inches='tight')
#     mplt.close()

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    eval_input: str = options.evaluation_file
    output: DH = DH(options.output_folder)

    sns.set_theme(context="paper")
    # mplt.style.use(['science','ieee'])

    # colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    colors = ["#e41a1c", "#377eb8"]
    cm = sns.color_palette(colors)
    font_file = '/home/mattia/src/SNNFrameWork/Fonts/helvetica.ttf'
    fm.fontManager.addfont(font_file)
    # font_manager.fontManager.ttflist.extend(font_list)
    # custom_params = {'figure.figsize':(8,8), "font.size": 44}
    custom_params = {"font.size": 34}
    sns.set_theme(rc=custom_params)
    # rcParams['figure.figsize'] = 8,4
    rcParams['font.size'] = 34
    # rcParams['axes.titlepad'] = 20

    y_limit = (69.8, 76.2)

    df = pd.read_csv(input)
    # df = df[df["epoch"] >= ep_limit]
    df["categorical_accuracy"] *= 100
    df["val_categorical_accuracy"] *= 100

    p = plt(df, format=["pdf", "png"])
    p.sns_set(font_scale=2.0)
    p.sns_set_api(sns.set_style, "white")
    p("line", x="epoch", y="categorical_accuracy", hue="function", palette=cm, linewidth = 2.0)
    p.set(title="Training accuracy evolution", xlabel="# Epoch", ylabel="Accuracy [%]", ylim=y_limit)
    p.set_legend(ncol=2, y=0.02, fontsize='18', title_fontsize='25')
    p.save(f"{output.path}/training_accuracy-presentation.pdf")

    p = plt(df, format=["pdf", "png"])
    p.sns_set(font_scale=2.0)
    p.sns_set_api(sns.set_style, "white")
    p("line", x="epoch", y="val_categorical_accuracy", hue="function", palette=cm, linewidth = 2.0)
    p.set(title="Validation accuracy evolution", xlabel="# Epoch", ylabel="Accuracy [%]", ylim=y_limit)
    p.set_legend(ncol=2, y=0.02, fontsize='18', title_fontsize='25')
    p.save(f"{output.path}/validation_accuracy-presentation.pdf")

    return 0

    # p = plt(df, format=["pdf", "png"])
    # p.sns_set(font_scale=2.25)
    # p.sns_set_api(sns.set_style, "white")
    # p("line", x="epoch", y="loss", hue="function", palette=cm, linewidth = 2.0)
    # p.set(xlabel="Epoch", ylabel="Loss", xlim=(ep_limit, None))
    # p.set_legend(ncol=2, y=0.02, fontsize='18', title_fontsize='25')
    # p.save(f"{output.path}/training_loss.pdf")

    # p = plt(df, format=["pdf", "png"])
    # p.sns_set(font_scale=2.25)
    # p.sns_set_api(sns.set_style, "white")
    # p("line", x="epoch", y="val_loss", hue="function", palette=cm, linewidth = 2.0)
    # p.set(xlabel="Epoch", ylabel="Loss", xlim=(ep_limit, None))
    # p.set_legend(ncol=2, y=0.02, fontsize='18', title_fontsize='25')
    # p.save(f"{output.path}/validation_loss.pdf")

    df = pd.read_csv(eval_input)
    df = df[df["Statistic"] == "categorical_accuracy"]
    # raincloud(df, ["CCE", "ACCE"], f"{output.path}/test_accuracy")

    # custom_params = {'figure.figsize':(8,4), "font.size": 34}
    custom_params = {"font.size": 44}
    sns.set_theme(rc=custom_params)
    # rcParams['figure.figsize'] = 8,4
    rcParams['font.size'] = 44
    # rcParams['axes.titlepad'] = 20

    df["Value"] *= 100.0
    p = plt(df, format=["pdf", "png"])
    p.sns_set(font_scale=1.5)
    p.sns_set_api(sns.set_style, "white")
    p("bar", x="function", y="Value", palette=cm)
    p.set(title="CCE VS ACCE Test accuracy comparison",
          xlabel=r"Loss function",
          ylabel=f"Accuracy [%]",
          ylim=(75.0, 76.0))
    # p.set_legend(ncol=3, y=0.05, fontsize='18', title_fontsize='25')
    p.save(f"{output.path}/accuracy_bar_plot.pdf")

#     df = df.round(3)
#     filter = ['TP', 'FP', 'TN', 'FN']
#     new_columns = ['VMAF', 'Origin', 'Evaluation', 'ValueTP', 'ValueFP', 'ValueTN', 'ValueFN', 'ValueAccuracy']
#     if multiExperiment:
#         new_columns.insert(2, 'Experiment')
#     if phase:
#         new_columns.insert(3, 'Phase')
#     # if phase:
#     #     # Prune not used phases
#     #     tmp_df = df[df["Phase"].isin(["NO_RL", "RL_Exploitation"])].copy()
#     #     tmp_df["Evaluation"] = np.where(tmp_df["Phase"].values == "RL_Exploitation", tmp_df["Evaluation"].values - 10, tmp_df["Evaluation"].values)
#     #     tmp_df.drop(["Phase"], axis=1, inplace=True)
#     #     df = tmp_df
#     #     print(df)
#     #     raise Exception

#     if undecided:
#         tmp_df = df[df["Phase"].isin(["NO_RL", "RL_Exploitation"])].copy()
#         tmp_df["Evaluation"] = list(range(0,10))*int(len(tmp_df)/10)
#         tmp_df = tmp_df[["VMAF", "Origin", "Evaluation", "Undecided"]].copy()
#         p = plt(tmp_df, format=["pdf", "png"])
#         p.sns_set_api(sns.set_style, rc={"axes.grid": True})
#         p("line", x="Evaluation", y="Undecided", hue="Origin", style="VMAF", markers=True)
#         p.set(title="Difficult samples # Undecided evolution",
#               xlabel="Evaluation", ylabel="# Undecided samples",
#               yscale="log", yticks=[100, 1000, 10000])
#         locmaj = mtkr.LogLocator(base=10, subs=np.arange(0.1,1,0.1), numticks=5)
#         p.plot.axes.yaxis.set_major_locator(locmaj)
#         p.plot.axes.yaxis.set_minor_formatter(mtkr.NullFormatter())
#         p.set_legend(ncol=2)

#         p.save(f"{output.path}/grays_undecided_evolution.pdf")

#         new_columns.append('ValueUndecided')
#         filter.append('Undecided')

#     if frozen:
#         tmp_df = df[["Origin", "Evaluation", "Frozen"]].copy()
#         p = plt(tmp_df, format=["pdf", "png"])
#         p("line", x="Evaluation", y="Frozen", hue="Origin", style="Origin", markers=True)
#         p.set(title="GrayInference Frozen samples evolution comparison",
#               xlabel="Evaluation", ylabel="# Frozen samples")
#         p.save(f"{output.path}/grays_frozen_evolution.pdf")

#         new_columns.append('ValueFrozen')

#     df.columns = new_columns
#     if multiExperiment:
#         df = to_long(df, stubnames='Value', i=['VMAF', 'Origin', 'Experiment', 'Phase', 'Evaluation'], j="Stat", suffix=".*")
#     else:
#         df = to_long(df, stubnames='Value', i=['VMAF', 'Origin', 'Evaluation'], j="Stat", suffix=".*")

#     sl_acc = df[(df["Origin"] == "SL") & (df["Stat"].isin(["Accuracy"]))].copy()
#     sl_p = df[(df["Origin"] == "SL") & (df["Stat"].isin(["TP", "FP"]))].copy()
#     sl_n = df[(df["Origin"] == "SL") & (df["Stat"].isin(["TN", "FN"]))].copy()
#     sl_u = df[(df["Origin"] == "SL") & (df["Stat"].isin(["Undecided"]))].copy()
#     sl_p = sl_p.groupby(["VMAF", "Origin", "Experiment", "Phase", "Evaluation"])["Value"].sum().reset_index()
#     sl_p["Stat"] = "Positive"
#     sl_n = sl_n.groupby(["VMAF", "Origin", "Experiment", "Phase", "Evaluation"])["Value"].sum().reset_index()
#     sl_n["Stat"] = "Negative"
#     sl_df = pd.concat([sl_p, sl_n, sl_u]).reset_index(drop=True)

#     # for vmaf in [80]:
#     for vmaf in sl_df["VMAF"].unique():
#         tmp_sl_df = sl_df[sl_df["VMAF"] == vmaf].copy()
#         # for exp in [3]:
#         for exp in tmp_sl_df["Experiment"].unique():
#             tmp_tmp_sl_df = tmp_sl_df[tmp_sl_df["Experiment"] == exp].copy()
#             p = plt(tmp_tmp_sl_df, format=["pdf", "png"])
#             p("lineStack", x="Evaluation", y="Value", hue="Stat", normalize=True)
#             p.set(xlabel="SL-Cycle",
#                   ylabel=f"Set distribution [0-1]",
#                   title="Dst. U, Labels distribution during SL process")
#             p.plot.set_xticks(range(0,9))
#             p.plot.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
#             p.set_legend(loc="upper left", x=1.03, y=1, ncol=1, title="Label")
#             p.save(f"{output.path}/DifficultEvolution-SL-{exp}-{vmaf}.pdf")

#     sl_p = df[(df["Origin"] == "SL-RL") & (df["Phase"] == "RL_Exploitation") & (df["Stat"].isin(["TP", "FP"]))].copy()
#     sl_n = df[(df["Origin"] == "SL-RL") & (df["Phase"] == "RL_Exploitation") & (df["Stat"].isin(["TN", "FN"]))].copy()
#     sl_u = df[(df["Origin"] == "SL-RL") & (df["Phase"] == "RL_Exploitation") & (df["Stat"].isin(["Undecided"]))].copy()
#     sl_p = sl_p.groupby(["VMAF", "Origin", "Experiment", "Phase", "Evaluation"])["Value"].sum().reset_index()
#     sl_p["Stat"] = "Positive"
#     sl_n = sl_n.groupby(["VMAF", "Origin", "Experiment", "Phase", "Evaluation"])["Value"].sum().reset_index()
#     sl_n["Stat"] = "Negative"
#     sl_df = pd.concat([sl_p, sl_n, sl_u]).reset_index(drop=True)

#     for vmaf in sl_df["VMAF"].unique():
#         tmp_sl_df = sl_df[sl_df["VMAF"] == vmaf].copy()
#         for exp in tmp_sl_df["Experiment"].unique():
#             tmp_tmp_sl_df = tmp_sl_df[tmp_sl_df["Experiment"] == exp].copy()
#             p = plt(tmp_tmp_sl_df, format=["pdf", "png"])
#             p("lineStack", x="Evaluation", y="Value", hue="Stat", normalize=True)
#             p.set(xlabel="SL-Cycle",
#                   ylabel=f"Set distribution [0-1]",
#                   title="Difficult samples labelling VS cycle")
#             p.plot.set_xticks(range(1,10))
#             # p.plot.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
#             p.set_legend(ncol=3, y=-0.42)
#             p.save(f"{output.path}/DifficultEvolution-SL-RL-{exp}-{vmaf}.pdf")


#     accuracy_df = df[(df["Stat"].isin(['Accuracy'])) & (df["Phase"].isin(["NO_RL"]))].copy()
#     accuracy_df["Value"] *= 100
#     accuracy_df["Evaluation"] = list(range(0,10))*int(len(accuracy_df)/10)
#     # for vmaf in accuracy_df["VMAF"].unique():
#     for vmaf in [90]:
#         tmp_acc_df = accuracy_df[accuracy_df["VMAF"] == vmaf].copy()
#         p = plt(tmp_acc_df, format=["pdf", "png"])
#         p.sns_set(font_scale=2.25)
#         p.sns_set_api(sns.set_style, "white")
#         p("line", x="Evaluation", y="Value", markers=True, linewidth=2.0, dashes=True)
#         p.set(xlabel="Evaluation cycle",
#               ylabel=f"Accuracy [%]")
#         # p.set_legend(ncol=2, y=-0.4)
#         p.save(f"{output.path}/grays_accuracy_evolution-VMAF{vmaf}.pdf")

#     colors = ["#e41a1c", "#377eb8", "#4daf4a"]
#     cm = sns.color_palette(colors)
#     p = plt(accuracy_df, format=["pdf", "png"])
#     p("line", x="Evaluation", y="Value", hue="VMAF", palette=cm, markers=True, dashes=True)
#     p.set(xlabel="Evaluation cycle",
#           ylabel=f"Accuracy [%]",
#           title="U samples Accuracy Evolution")
#     p.set_legend(ncol=3, y=-0.5)
#     p.save(f"{output.path}/grays_accuracy_evolution.pdf")

#     df = pd.read_csv(evolution_file)
#     df.columns = ["VMAF", "Origin", "Experiment", "Phase", "epoch", "Evaluation", "ValueAccuracy", "ValueLoss", "ValueValAccuracy", "ValueValLoss"]
#     df = to_long(df, stubnames='Value', i=['VMAF', 'Origin', 'Experiment', 'Phase', "epoch", 'Evaluation'], j="Stat", suffix=".*")

#     # tmp_df = df[(df["Stat"].isin(["Accuracy", "ValAccuracy"])) & (df["Phase"].isin(["NO_RL", "RL_Exploitation"]))]
#     # for evaluation in tmp_df["Evaluation"].unique():
#     #     acc_df = tmp_df[tmp_df["Evaluation"] == evaluation]
#     #     p = plt(acc_df, format=["pdf", "png"])
#     #     p("line", x="epoch", y="Value", hue="Origin", style="Stat", markers=True, dashes=True)
#     #     p.set(xlabel="Epoch",
#     #           ylabel=f"Accuracy value",
#     #           title=f"Gray-Evaluation {evaluation} accuracy evolution")
#     #     p.set_legend(ncol=2, y=-0.5)
#     #     p.save(f"{output.path}/accuracy_evolution_evolution{evaluation}.pdf")

#     # Last evaluation confusion matrix boxplot
#     # cm = df[df["Stat"].isin(filter)].copy()
#     # last_eval = max(cm["Evaluation"].unique())
#     # cm_last_eval = cm[cm["Evaluation"] == last_eval].copy()
#     # sum_df = cm_last_eval.groupby(["Origin", "Experiment"])["Value"].sum().reset_index()
#     # assert len(sum_df["Value"].unique()) == 1
#     # total = sum_df["Value"].unique()[0]
#     # cm_last_eval["Value"] /= total

#     # p = plt(cm_last_eval, format=["pdf", "png"])
#     # p("violin", x="Stat", y="Value", hue="Origin", split=True, cut=0)
#     # p.set(xlabel="Confusion matrix label", ylabel=f"Value [0-1]",title="Difficult set confusion matrix last cycle")
#     # p.set_legend(ncol=2)
#     # p.save(f"{output.path}/grays_cm_last_cycle.pdf")

#     # conf_matrix = df[df["Stat"].isin(['TP', 'FP', 'TN', 'FN', 'Undecided'])].copy()
#     # conf_matrix["Value"] /= 14264
#     # p = plt(conf_matrix, format=["pdf", "png"])
#     # p("lineStack", x="Evaluation", y="Value", hue="Stat")
#     # p.set(xlabel="Evaluation", ylabel=f"Value Ratio [0-1]", title="GrayInference Confusion Matrix Evolution")
#     # p.set_legend(ncol=3, y=-0.42)
#     # p.save(f"{output.path}/grays_confusion_matrix_evolution.pdf")

#     if evaluation_file is not None and multiExperiment:
#         evaluation_df = pd.read_csv(evaluation_file)
#         accuracy_eval_df = evaluation_df[(evaluation_df["Statistic"] == "MNO_Accuracy") | (evaluation_df["Statistic"] == "OTT_Accuracy")]
#         accuracy_eval_df["Value"] *= 100
#         identifier = ["0", "1", "2"]
#         # identifier = ["0", "1"]
#         MNO_cm_labels = [f"{s}_{o}_{v}" for s, o, v in product(["MNO"], identifier, ["TP", "FP", "TN", "FN", "U"])]
#         OTT_cm_labels = [f"{s}_{o}_{v}" for s, o, v in product(["OTT"], identifier, ["TP", "FP", "TN", "FN", "U"])]
#         MNO_expected_cm_labels = [f"{s}_{o}_expected_{v}" for s, o, v in product(["MNO"], identifier, ["P", "N"])]
#         OTT_expected_cm_labels = [f"{s}_{o}_expected_{v}" for s, o, v in product(["OTT"], identifier, ["P", "N"])]
#         MNO_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(MNO_cm_labels)].copy()
#         MNO_cm_eval_df["Statistic"] = MNO_cm_eval_df["Statistic"].str.replace('MNO_.*_',"MNO_0_", regex=True)
#         MNO_cm_eval_df = MNO_cm_eval_df.copy().groupby(["VMAF", "Origin", "Experiment", "Statistic"])["Value"].sum().reset_index()
#         # OTT_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(OTT_cm_labels)].copy()
#         MNO_expected_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(MNO_expected_cm_labels)].copy()
#         MNO_expected_cm_eval_df["Statistic"] = MNO_expected_cm_eval_df["Statistic"].str.replace('MNO_.*_',"MNO_0_", regex=True)
#         MNO_expected_cm_eval_df = MNO_expected_cm_eval_df.copy().groupby(["VMAF", "Origin", "Experiment", "Statistic"])["Value"].sum().reset_index()

#         # OTT_expected_cm_eval_df = evaluation_df[evaluation_df["Statistic"].isin(OTT_expected_cm_labels)].copy()

#         def separate_org_cm(df: pd.DataFrame) -> pd.DataFrame:
#             splt = df['Statistic'].str.split('_(.*?)_', expand=True)
#             df = pd.concat([df, splt], axis=1)
#             return df

#         def normalize_cm(df, exp_df) -> pd.DataFrame:
#             for i in set(exp_df.index):
#                 sub_exp_df = exp_df.loc[i]
#                 p = sub_exp_df[sub_exp_df["Label"] == "expected_P"].values[0][0]
#                 n = sub_exp_df[sub_exp_df["Label"] == "expected_N"].values[0][0]
#                 sub_df = df.loc[i].copy()
#                 p_mask = (sub_df["CM"] == "TP") | (sub_df["CM"] == "FN")
#                 n_mask = (sub_df["CM"] == "TN") | (sub_df["CM"] == "FP")
#                 u_mask = sub_df["CM"] == "U"
#                 sub_df.loc[p_mask, "Value"] = (sub_df.loc[p_mask]["Value"] / p)
#                 sub_df.loc[n_mask, "Value"] = (sub_df.loc[n_mask]["Value"] / n)
#                 sub_df.loc[u_mask, "Value"] = (sub_df.loc[u_mask]["Value"] / (p+n))
#                 df.loc[i] = sub_df.loc[i]
#             df.fillna(0, inplace=True)
#             return df

#         def cm_pp(df: pd.DataFrame, expected_df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
#             df = separate_org_cm(df)
#             exp_df = separate_org_cm(expected_df)
#             df.rename({'Origin': "Approach", 0: 'Side', 1: 'Origin', 2: 'CM'}, axis=1, inplace=True)
#             exp_df.rename({'Origin': "Approach", 0: 'Side', 1: 'Origin', 2: 'Label'}, axis=1, inplace=True)
#             df.drop(["Statistic", "Side"], axis=1, inplace=True)
#             exp_df.drop(["Statistic", "Side"], axis=1, inplace=True)
#             df = df.replace({'Origin': "0"}, value="Positive")
#             df = df.replace({'Origin': "1"}, value="Negative")
#             df = df.replace({'Origin': "2"}, value="Difficult")
#             df = df.set_index(["Experiment", "Approach", "Origin"])
#             exp_df = exp_df.replace({'Origin': "0"}, value="Positive")
#             exp_df = exp_df.replace({'Origin': "1"}, value="Negative")
#             exp_df = exp_df.replace({'Origin': "2"}, value="Difficult")
#             exp_df = exp_df.set_index(["Experiment", "Approach", "Origin"])
#             if normalize:
#                 df = normalize_cm(df, exp_df)
#             return df

#         def get_cm_stats(df, a, o, s):
#             pcr = [.25, .5, .75]
#             tp = df[df["CM"] == "TP"]["Value"].describe(percentiles=pcr)
#             fp = df[df["CM"] == "FP"]["Value"].describe(percentiles=pcr)
#             tn = df[df["CM"] == "TN"]["Value"].describe(percentiles=pcr)
#             fn = df[df["CM"] == "FN"]["Value"].describe(percentiles=pcr)
#             u = df[df["CM"] == "U"]["Value"].describe(percentiles=pcr)
#             stat = pd.concat([tp, fp, tn, fn, u], axis=1)
#             stat.columns = ["TP", "FP", "TN", "FN", "U"]
#             stat = stat.T.copy()
#             stat = stat.round(3)
#             stat["Statistic"] = ["TP", "FP", "TN", "FN", "U"]
#             stat["Side"] = s
#             stat["Approach"] = a
#             stat["Origin"] = o
#             stat.drop(["count"], axis=1, inplace=True)
#             stat.reset_index(inplace=True, drop=True)
#             return stat

#         def calculate_stats(df, s) -> pd.DataFrame:
#             df_stats = pd.DataFrame()
#             for ap in df["Approach"].unique():
#                 tmp_df = df[df["Approach"] == ap]
#                 for org in tmp_df["Origin"].unique():
#                     tmp_tmp_df = tmp_df[tmp_df["Origin"] == org]
#                     tmp_tmp_st = get_cm_stats(tmp_tmp_df, ap, org, s)
#                     df_stats = pd.concat([df_stats, tmp_tmp_st])
#             df_stats.drop(["min", "max"], axis=1, inplace=True)
#             df_stats["Mean (std)"] = df_stats["mean"].astype(str) + " (" + df_stats["std"].astype(str) + ")"
#             df_stats.drop(["mean", "std"], axis=1, inplace=True)
#             df_stats.set_index(["Side", "Approach", "Origin", "Statistic"], inplace=True)
#             cols = df_stats.columns.tolist()
#             cols = cols[-1:] + cols[:-1]
#             df_stats = df_stats[cols]
#             return df_stats

#         def plot_cm(df: pd.DataFrame, label: str) -> None:
#             df.replace("Positive", "Trivial-Positive", inplace=True)
#             df.replace("Negative", "Trivial-Negative", inplace=True)
#             out = f"{output.path}/{label}_cm_plot_noNorm.pdf"
#             p = plt(df, format=["pdf", "png"])
#             # hue_order_labels = ["Trivial", "SL"]
#             hue_order_labels = ["Baseline", "SL", "SL-RL"]
#             # hue_order_labels = ["Def0.5", "Random"]
#             # hue_order_labels = ["Def0.5"]
#             # hue_order_labels = ["Trivial", "SL", "SL-RL", "Def0.5", "Keep", "Random"]
#             p("cat", x="CM", y="Value", hue="Approach", col="Origin", row="CM", kind="box", sharey=False, sharex=False, margin_titles=True, row_order=["TP", "FP", "TN", "FN", "U"], hue_order=hue_order_labels, legend=False)
#             p.sns_set_api(interface=p.plot.set_titles, col_template="Dataset {col_name}", row_template="CM label {row_name}")
#             p.sns_set_api(interface=p.plot.set_axis_labels, x_var="Confusion matrix Labels", y_var="# Samples")
#             p.set_legend(ncol=3, x=-0.55, y=-0.4, title="Approach")
#             p.save(out)

#         def define_cm(df, expected_df, s, plot: bool =True) -> None:
#             cm = cm_pp(df, expected_df)
#             cm = cm.reset_index(drop=False)
#             cm.drop(["Experiment"], axis=1, inplace=True)
#             cm_norm = cm_pp(df, expected_df, normalize=False)
#             cm_norm = cm_norm.reset_index(drop=False)
#             cm_norm.drop(["Experiment"], axis=1, inplace=True)
#             if plot:
#                 plot_cm(cm_norm, s)
#             cm_stats = calculate_stats(cm, s)
#             cm_stats.to_csv(f"{output.path}/{s}_cm.csv")
#             cm_stats.rename({'25%': "25\\%", "50%": '50\\%', "75%": '75\\%'}, axis=1, inplace=True)
#             cm_stats[["25\\%","50\\%","75\\%"]] = cm_stats[["25\\%","50\\%","75\\%"]].round(3)
#             cm_stats[["25\\%","50\\%","75\\%"]] = cm_stats[["25\\%","50\\%","75\\%"]].astype(str)

#             tex_table = cm_stats.style.to_latex(column_format="|l|l|l|l|l|r|r|r|",
#                                                 position_float="centering",
#                                                 multicol_align="|c|",
#                                                 hrules=True)
#             txt = f"""\\documentclass{{article}}
# \\usepackage{{multirow}}
# \\usepackage{{booktabs}}

# \\title{{tmp}}
# \\author{{Mattia Milani}}
# \\date{{February 2023}}

# \\begin{{document}}

# {tex_table}

# \\end{{document}}
# """
#             with open(f"{output.path}/{s}_cm.tex", 'w+') as f:
#                 f.write(txt)
#             proc = subprocess.Popen(['pdflatex', f"-output-directory={output.path}",f"{output.path}/{s}_cm.tex"])
#             proc.communicate()
#             return cm_stats


#         # np.testing.assert_array_equal(MNO_cm_eval_df["Value"].values, OTT_cm_eval_df["Value"].values)
#         if "VMAF" in MNO_cm_eval_df.columns:
#             final_df = pd.DataFrame()
#             for v in MNO_cm_eval_df["VMAF"].unique():
#                 tmp_eval_df = MNO_cm_eval_df[MNO_cm_eval_df["VMAF"] == v].copy()
#                 tmp_exp_df = MNO_expected_cm_eval_df[MNO_expected_cm_eval_df["VMAF"] == v].copy()
#                 tmp_df = define_cm(tmp_eval_df, tmp_exp_df, f"MNO-{v}", plot=False)
#         # define_cm(OTT_cm_eval_df, OTT_expected_cm_eval_df, "OTT")

#         MNO_cm_eval_df["Statistic"] = MNO_cm_eval_df["Statistic"].str.replace('MNO_.*_',"", regex=True)
#         p_cm_df = MNO_cm_eval_df[MNO_cm_eval_df["Statistic"].isin(["TP", "FP"])]
#         tp_cm_df = MNO_cm_eval_df[MNO_cm_eval_df["Statistic"].isin(["TP"])].copy()
#         total_p_cm_df = p_cm_df.groupby(["VMAF", "Origin", "Experiment"])["Value"].sum().reset_index()
#         total_p_cm = total_p_cm_df["Value"].values
#         tp_cm_df["Value"] = (tp_cm_df["Value"].values/total_p_cm)*100
#         tp_cm_df["Statistic"] = "Precision"
#         precision_df = tp_cm_df.copy()

#         def raincloud(df, ticks, output_file,
#                       title: str = "Test-Dataset Accuracy",
#                       ylabel: str = "Accuracy [%]",
#                       xlabel: str = "VMAF Threshold",
#                       **kwargs):
#             raincloud_kw = {
#                         "data": df,
#                         "x": "Origin",
#                         "y": "Value",
#                         "palette": "Set2",
#                         "bw": 0.2,
#                         "width_viol": .5,
#                         "alpha": .5,
#                         "dodge": True,
#                         "move": 0.2,
#                         "pointplot": True,
#                         "box_medianprops": {"linewidth": 1., "zorder": 10}
#                     }
#             raincloud_kw.update(kwargs)

#             # ax = pt.RainCloud(**raincloud_kw)
#             # ax.set_xticklabels(ticks)
#             # mplt.title(title)
#             # mplt.ylabel(ylabel)
#             # mplt.xlabel(xlabel)
#             # # handles, labels = ax.get_legend_handles_labels()
#             # # ax.get_legend().remove()
#             # # mplt.legend(handles[:len(df["Statistic"].unique())],
#             # #             labels[:len(df["Statistic"].unique())],
#             # #             bbox_to_anchor=(0.5, -0.40),
#             # #             loc="lower center", ncol=2, borderaxespad=0,
#             # #             title="Accuracy")
#             # mplt.savefig(f"{output.path}/{output_file}.pdf", bbox_inches='tight')
#             # mplt.savefig(f"{output.path}/{output_file}.png", bbox_inches='tight')
#             # mplt.close()

#             raincloud_kw.update({
#                     "x": "VMAF",
#                     "hue": "Origin",
#                     "pointplot": True
#                 })
#             ax = pt.RainCloud(**raincloud_kw)
#             ax.set_xticklabels(df["VMAF"].unique())
#             mplt.title(title)
#             # ax.set_ylim((0.90, 1.0))
#             mplt.ylabel(ylabel)
#             mplt.xlabel(xlabel)
#             handles, labels = ax.get_legend_handles_labels()
#             ax.get_legend().remove()
#             mplt.legend(handles[:len(df["Origin"].unique())],
#                         labels[:len(df["Origin"].unique())],
#                         bbox_to_anchor=(0.5, -0.41),
#                         loc="lower center", ncol=3, borderaxespad=0,
#                         title="Approach")
#             mplt.savefig(f"{output.path}/{output_file}_reverse.pdf", bbox_inches='tight')
#             mplt.savefig(f"{output.path}/{output_file}_reverse.png", bbox_inches='tight')
#             mplt.close()

#         # raincloud(accuracy_eval_df, ["GB-Default", "SL", "10Ep", "GreyCycle", "FixedM"], "test_accuracy_comparison_all")
#         # default = accuracy_eval_df[accuracy_eval_df["Origin"] == "Default"]
#         # raincloud(default, ["GoodBad"], "test_accuracy_comparison_default_only")
#         # default_sl = accuracy_eval_df[(accuracy_eval_df["Origin"] == "Default") | (accuracy_eval_df["Origin"] == "SL")]
#         # raincloud(default_sl, ["GoodBad", "SelfLearning"], "test_accuracy_comparison_default_SL")
#         # tr_l = "Def0.5"
#         # sl_l = "Keep"
#         # slrl_l = "Random"
#         # origin = ["Trivial", "SL", "SL-RL", "Def0.5", "Keep", "Random"]
#         # origin = ["Trivial", "SL"]
#         origin = ["Baseline", "SL", "SL-RL"]
#         # origin = ["Def0.5", "Random"]
#         # origin = ["Def0.5"]

#         default_sl = accuracy_eval_df[accuracy_eval_df["Origin"].isin(origin)]
#         default_sl = default_sl[default_sl["Statistic"] == "MNO_Accuracy"]
#         default_sl["Statistic"] = "Accuracy"
#         default_sl["Origin"] = default_sl['Origin'].str.replace('SL-RL','ATELIER')
#         precision_df["Origin"] = precision_df['Origin'].str.replace('SL-RL','ATELIER')

#         # default_sl = accuracy_eval_df[(accuracy_eval_df["Origin"] == tr_l) | \
#         #                               (accuracy_eval_df["Origin"] == sl_l)]
#         # trivial = default_sl[default_sl["Origin"] == tr_l]
#         # sl = default_sl[default_sl["Origin"] == sl_l]
#         # slrl = default_sl[default_sl["Origin"] == slrl_l]
#         # print(trivial)
#         # raise Exception
#         # tr__stat = trivial["Accuracy"]["Value"].describe(percentiles=[.1, .25, .5, .75, .9])
#         # tr_ott_stat = trivial[trivial["Statistic"] == "OTT_Accuracy"]["Value"].describe(percentiles=[.1, .25, .5, .75, .9])
#         # sl_stat = sl["Accuracy"]["Value"].describe(percentiles=[.1, .25, .5, .75, .9])
#         # sl_ott_stat = sl[sl["Statistic"] == "OTT_Accuracy"]["Value"].describe(percentiles=[.1, .25, .5, .75, .9])
#         # sr_mno_stat = slrl[slrl["Statistic"] == "MNO_Accuracy"]["Value"].describe(percentiles=[.1, .25, .5, .75, .9])
#         # sr_ott_stat = slrl[slrl["Statistic"] == "OTT_Accuracy"]["Value"].describe(percentiles=[.1, .25, .5, .75, .9])
#         # stat = pd.concat([tr_mno_stat, tr_ott_stat, sl_mno_stat, sl_ott_stat, sr_mno_stat, sr_ott_stat], axis=1)
#         # stat.columns = ["Trivial_MNO", "Trivial_OTT", "SL_MNO", "SL_OTT", "SLRL_MNO", "SLRL_OTT"]
#         # stat = stat.T.copy()
#         # stat = stat.round(3)
#         # stat["Type"] = ["MNO", "OTT", "MNO", "OTT", "MNO", "OTT"]
#         # stat["Exp"] = [tr_l, tr_l, "Self Learning", "Self Learning", "SL+RL", "SL+RL"]
#         # stat.reset_index(inplace=True, drop=True)
#         # stat.to_csv(f"{output.path}/accuracy_stats.csv", index=False)
#         # raincloud(default_sl, [tr_l, "SelfLearning"], "test_accuracy_comparison_default_SL")

#         rain_cloud_title = f"Video {video_title} "+r"$D_{test}$ Accuracy comparison"
#         precision_title = f"Video {video_title} "+r"$D_{test}$ Precision comparison"
#         raincloud(default_sl, origin, "test_accuracy_comparison_default_SL", title=rain_cloud_title)

#         print(default_sl[(default_sl["VMAF"] == 99) & (default_sl["Origin"] == "ATELIER")]["Value"].mean())
#         print(default_sl[(default_sl["VMAF"] == 99) & (default_sl["Origin"] == "Baseline")]["Value"].mean())

#         p = plt(default_sl, format=["pdf", "png"])
#         p.sns_set(font_scale=2.25)
#         p.sns_set_api(sns.set_style, "white")
#         p("bar", x="VMAF", y="Value", hue="Origin", palette=cm)
#         p.set(xlabel=r"$v_{min}$",
#               ylabel=f"Accuracy [%]",
#               ylim=(0.0, 100.01))
#         p.set_legend(ncol=3, y=0.05, fontsize='18', title_fontsize='25')
#         p.save(f"{output.path}/accuracy_bar_plot.pdf")

#         p = plt(precision_df, format=["pdf", "png"])
#         p.sns_set(font_scale=2.25)
#         p.sns_set_api(sns.set_style, "white")
#         p("bar", x="VMAF", y="Value", hue="Origin", palette=cm)
#         p.set(xlabel=r"$v_{min}$", ylim=(0.0, 100.01), ylabel=r"Precision $\frac{TP}{TP+FP}$ [%]")
#         p.set_legend(ncol=3, y=0.05, fontsize='18', title_fontsize='25')
#         p.save(f"{output.path}/precision_bar_plot.pdf")
#         # raincloud(TP_MNO_OTT, [tr_l, "SelfLearning", "SL+RL"], "TP_MNO_OTT", title="TP test dst. Confusion Matrix (Higher better)", ylabel="% obtained over expected [0-1]", xlabel="Side")
#         # raincloud(FP_MNO_OTT, [tr_l, "SelfLearning", "SL+RL"], "FP_MNO_OTT", title="FP test dst. Confusion Matrix (Lower better)", ylabel="% obtained over expected [0-1]", xlabel="Side")
#         # raincloud(TN_MNO_OTT, [tr_l, "SelfLearning", "SL+RL"], "TN_MNO_OTT", title="TN test dst. Confusion Matrix (Higher better)", ylabel="% obtained over expected [0-1]", xlabel="Side")
#         # raincloud(FN_MNO_OTT, [tr_l, "SelfLearning", "SL+RL"], "FN_MNO_OTT", title="FN test dst. Confusion Matrix (Lower better)", ylabel="% obtained over expected [0-1]", xlabel="Side")
#         # raincloud(U_MNO_OTT, [tr_l, "SelfLearning", "SL+RL"], "U", title="U test dst. Confusion Matrix (Lower better)", ylabel="% obtained over expected [0-1]", xlabel="Side")
#         # raincloud(cm_eval_df, [tr_l, "SelfLearning", "SL+RL"], "test_cm_comparison")
#         # raincloud(MNO_cm_eval_df, ["Trivial", "SelfLearning", "SL+RL"], "test_MNO_cm_comparison")
#         # raincloud(OTT_cm_eval_df, ["Trivial", "SelfLearning", "SL+RL"], "test_OTT_cm_comparison")
#         p = plt(default_sl, format=["pdf", "png"])
#         p("displot", x="Value", col="Statistic", hue="Origin", kind="ecdf")
#         p.save(f"{output.path}/accuracy_ecdf.pdf")


if __name__ == "__main__":
    main()
