import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List

parser = argparse.ArgumentParser(usage="usage: plot.py [options]",
                                 description="Use the script to generate the plots from the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/statistics.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-g", "--grayFle", dest="gray_csv", default=None,
                    action="store", help="define the gray csv input file")
parser.add_argument("-u", "--undecided", dest="undecided", default=False,
                    action="store_true", help="Activate the flag for undecided gray samples")
parser.add_argument("--frozen", dest="frozen", default=False,
                    action="store_true", help="Activate the flag for frozen gray samples")
parser.add_argument("-m", "--multiIterationFile", dest="multiIteration", default=None,
                    action="store", help="define the multi iteration csv input file")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")

def save_plot(p, output: str, columns: int = 2, distance: float = .2, legend: bool = True) -> None:
    if legend:
        sns.move_legend(
            p, "upper center",
            bbox_to_anchor=(.5, -distance), ncol=columns, title=None,
        )
    p.figure.savefig(output, bbox_inches="tight")
    p.figure.savefig(output.replace('pdf','png'), bbox_inches="tight")
    plt.clf()

def plot_loss(df, output, title="Loss Evolution") -> None:
    loss_df = df[df["Stat"].isin(['Loss', 'Val_Loss'])]
    p = sns.lineplot(data=loss_df, x="Epoch", y="Value", hue="Stat", style="Approach",
                     hue_order=['Loss', 'Val_Loss'], style_order=['Trivial', 'SL'], markers=False, dashes=True)
    p.set(xlabel="Epoch", ylabel=f"Loss value")
    p.set(title=title)
    save_plot(p, output, columns=2)

def plot_accuracy(df, output, title="Accuracy Evolution") -> None:
    accuracy_df = df[df["Stat"].isin(['Accuracy'])]
    print(accuracy_df)
    p = sns.lineplot(data=accuracy_df, x="Epoch", y="Value", hue="Approach",
                     hue_order=['Trivial', 'SL'], markers=False, dashes=True)
    p.set(xlabel="Epoch", ylabel=f"Accuracy value")
    p.set(title=title)
    save_plot(p, output, columns=2)

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    output: str = options.output_folder
    grays: str = options.gray_csv
    multiIteration: str = options.multiIteration
    undecided: bool = options.undecided
    frozen: bool = options.frozen

    def to_long(df, **kwargs):
        tmp_df = pd.wide_to_long(df, **kwargs)
        tmp_df.reset_index(inplace=True)
        return tmp_df

    # df = pd.read_csv(input)
    # df.columns = ['Epoch',
    #               'ValueAccuracy',
    #               'ValueLoss',
    #               'ValueVal_Accuracy',
    #               'ValueVal_Loss']
    # df = to_long(df, stubnames='Value', i=['Epoch'], j="Stat", suffix=".*")

    # Seaborn settings
    sns.set(font_scale=1.5)
    sns.set_style("white")
    sns.set_palette(sns.color_palette())

    # plot_loss(df, f"{output}loss_evolution.pdf")
    # plot_accuracy(df, f"{output}accuracy_evolution.pdf")

    if multiIteration is not None:
        df = pd.read_csv(multiIteration)
        print(df)
        df.columns = ['Iteration',
                      'Approach',
                      'Epoch',
                      'ValueAccuracy',
                      'ValueLoss',
                      'ValueVal_Accuracy',
                      'ValueVal_Loss']
        df = to_long(df, stubnames='Value', i=['Iteration', 'Approach', 'Epoch'], j="Stat", suffix=".*")

        # plot_loss(df, f"{output}gray_loss_evolution.pdf", title="SelfLearning approach 2-10 iteration Loss evolution")
        # plot_accuracy(df, f"{output}gray_accuracy_evolution.pdf", title="SelfLearning approach 2-10 iteration Accuracy evolution")

        for val in df["Iteration"].unique():
            tmp_df = df[df["Iteration"] == val]
            plot_loss(tmp_df, f"{output}gray_loss_evolution_iteration_{val}.pdf", title=f"SelfLearning approach iteration {val} Loss evolution")
            plot_accuracy(tmp_df, f"{output}gray_accuracy_evolution_iteration_{val}.pdf", title=f"Iteration {val} Train Accuracy evolution")

    raise Exception

    if grays is not None:
        df = pd.read_csv(grays)
        df["Evaluation"] = range(1, len(df.index)+1)
        df = df.round(3)
        filter = ['TP', 'FP', 'TN', 'FN']
        new_columns = ['ValueTP', 'ValueFP', 'ValueTN', 'ValueFN', 'ValueAccuracy']
        if undecided:
            tmp_df = df[["Evaluation", "Undecided"]].copy()
            p = sns.lineplot(data=tmp_df, x="Evaluation", y="Undecided")
            p.set(xlabel="Evaluation", ylabel=f"# Undecided samples")
            p.set(title="Undefined sub-set undecided samples evolution")
            save_plot(p, f"{output}grays_undecided_evolution.pdf", legend=False)

            new_columns.append('ValueUndecided')
            filter.append('Undecided')

        if frozen:
            tmp_df = df[["Evaluation", "Frozen"]].copy()
            p = sns.lineplot(data=tmp_df, x="Evaluation", y="Frozen")
            p.set(xlabel="Evaluation", ylabel=f"# Frozen samples")
            p.set(title="Undefined sub-set frozen samples evolution")
            save_plot(p, f"{output}grays_frozen_evolution.pdf", legend=False)

            new_columns.append('ValueFrozen')

        new_columns.append("Evaluation")
        df.columns = new_columns
        df = to_long(df, stubnames='Value', i=['Evaluation'], j="Stat", suffix=".*")

        accuracy_df = df[df["Stat"].isin(['Accuracy'])]
        p = sns.lineplot(data=accuracy_df, x="Evaluation", y="Value", markers=False, dashes=True)
        p.set(xlabel="Evaluation", ylabel=f"Accuracy value")
        p.set(title="Undefined sub-set Accuracy Evolution")
        save_plot(p, f"{output}grays_accuracy_evolution.pdf", legend=False)

        conf_matrix = df[df["Stat"].isin(['TP', 'FP', 'TN', 'FN'])]
        p = sns.lineplot(data=conf_matrix, x="Evaluation", y="Value", hue="Stat", style="Stat",
                         hue_order=['TP', 'FP', 'TN', 'FN'], style_order=['TP', 'FP', 'TN', 'FN'], markers=False, dashes=True)
        p.set(xlabel="Evaluation", ylabel=f"Value Ratio [0-1]")
        p.set(title="Undefined sub-set Confusion Matrix Evolution")
        save_plot(p, f"{output}grays_confusion_matrix_evolution.pdf", columns=4)

        def expand(df: pd.DataFrame, id_col: str = "", val_col: str = "", hue_col: str = "") -> pd.DataFrame:
            res = pd.DataFrame(columns=[id_col, hue_col])
            for id in df[id_col].unique():
                tmp_df = df[df[id_col] == id]
                for stat in tmp_df[hue_col].unique():
                    reps = tmp_df[tmp_df[hue_col] == stat][val_col].values[0]
                    tmp_res = pd.DataFrame({
                            id_col: id,
                            hue_col: np.repeat(stat, reps)
                        })
                    res = pd.concat([res, tmp_res])
            return res

        conf_matrix = expand(df[df["Stat"].isin(filter)], "Evaluation", "Value", "Stat")
        conf_matrix = conf_matrix.reset_index(drop=True)
        p = sns.kdeplot(data=conf_matrix, x="Evaluation", hue="Stat", bw_adjust=2., multiple="fill", hue_order=filter)
        p.set(xlim=(0, max(conf_matrix["Evaluation"].unique())))
        p.set(xlabel="Evaluation")
        p.set(title="Undefined sub-set Confusion Matrix Evolution")
        save_plot(p, f"{output}grays_confusion_matrix_distEvolution.pdf", columns=3)

if __name__ == "__main__":
    main()
