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

import ast
from copy import deepcopy
from tqdm import tqdm
import math
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as mplt
import umap
import numpy as np
import pandas as pd
import tensorflow as tf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.lines import Line2D
from SNN2.src.decorators.decorators import c_logger
from SNN2.src.util.helper import dst2tensor

from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.files import FileHandler as FH
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.core.data.PreProcessing import PreProcessing as PP
from SNN2.src.plot.plotter import plotter as plt
from sklearn.preprocessing import StandardScaler
from SNN2.src.io.progressBar import pb

from typing import Dict, List, Optional, Tuple, Generator, Any

class Transform:
    """Transform.
    Class that applies general transformations to data
    """

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


@c_logger
class Study:

    def __init__(self,
                 params: PH,
                 pp: PP,
                 action_param: PH,
                 ph: PkH,
                 hash: str = ""):
        # Basic objects to generate an experiment
        self.ph: PkH = ph
        self.params: PH = params
        self.data: PP = pp
        self.actions: PH = action_param
        self.hash = hash
        self.stat_xlims = {
                    "PDR": (-.5, .5),
                    "BDR": (-.5, None),
                    "AvgIPTD": (None, None),
                    "StdIPTD": (None, None),
                    "SkwIPTD": (None, 6.),
                    "KurIPTD": (None, 50),
                    "VMAF": (None, None)
                }
        self.stat_columns = ["ValuePDR",
                             "ValueBDR",
                             "ValueAvgIPTD",
                             "ValueStdIPTD",
                             "ValueSkwIPTD",
                             "ValueKurIPTD"]
        self.output = self.params["studyOutput"]

        self.data_dst()
        raise Exception
        # self.feature_correlation()
        # self.similarity_study()
        # self.average_windows_study()
        # self.target_distribution_study()
        # self.false_positives_ipothesis()

    def to_long(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        # df = pd.wide_to_long(df, stubnames="Value", i=["i"], j="Stat", suffix=".*").reset_index()
        df['index'] = df.index
        df = pd.wide_to_long(df, *args, i="index", suffix=".*", **kwargs).reset_index()
        df.drop(["index"], axis=1, inplace=True)
        return df

    def windows2df(self, wdw: tf.Tensor) -> pd.DataFrame:
        self.write_msg(f"wdws shapes: {wdw.shape}")
        wdw_np = np.reshape(wdw.numpy(), (wdw.shape[0]*wdw.shape[1], wdw.shape[2]))
        samples_per_window = wdw.shape[1]
        num_windows = wdw.shape[0]
        wdw_id = np.repeat(np.arange(num_windows), samples_per_window)
        df = pd.DataFrame(wdw_np, columns=self.stat_columns)
        df["id"] = wdw_id
        return df

    def plot(self, plotter: plt,
             plot_function: str,
             *args,
             set_kwargs: Optional[Dict[str, str]] = None,
             legend_kwargs: Optional[Dict[str, str]] = None,
             **kwargs
             ) -> None:
        if set_kwargs is None:
            set_kwargs = {}
        if legend_kwargs is None:
            legend_kwargs = {}

        plotter(plot_function, *args, **kwargs)
        plotter.set(**set_kwargs)
        plotter.set_legend(**legend_kwargs, fontsize='18')

    def pdf(self, df: pd.DataFrame,
            output: str,
            *args,
            **kwargs) -> None:
        p = plt(df, format=["pdf", "png"])
        self.plot(p, "kde", *args, **kwargs)
        p.save(output)

    def ecdf(self, df: pd.DataFrame,
             output: str,
             *args,
             palette: Optional[Any] = None,
             **kwargs) -> None:
        df.drop([0], axis=1, inplace=True)
        df.dropna(inplace=True)
        p = plt(df, format=["pdf", "png"], palette=palette)
        p.sns_set(font_scale=2.25)
        p.sns_set_api(sns.set_style, "white")
        self.plot(p, "ecdf", *args, **kwargs)
        for lines, linestyle, legend_handle in zip(p.plot.axes.lines, [':', '--', '-'], p.plot.axes.legend_.legendHandles):
            lines.set_linestyle(linestyle)
            legend_handle.set_linestyle(linestyle)
        mplt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0', '25', '50', '75', '100'])
        p.save(output)

    def line(self, df: pd.DataFrame,
             output: str,
             *args,
             palette: Optional[Any] = None,
             **kwargs) -> None:
        p = plt(df, format=["pdf", "png"], palette=palette)
        p.sns_set(font_scale=2.25)
        p.sns_set_api(sns.set_style, "white")
        self.plot(p, "line", *args, **kwargs)
        self.write_msg(len(p.plot.axes.lines))
        self.write_msg(len(p.plot.axes.legend_.legendHandles))
        lines = p.plot.axes.lines[0:3]
        self.write_msg(len(lines))
        for lines, linestyle, legend_handle, color in zip(lines, [':', '--', '-'], p.plot.axes.legend_.legendHandles, palette):
            lines.set_linestyle(linestyle)
            if linestyle == '--':
                self.write_msg(f"handle type: {type(legend_handle)}")
                legend_handle.set_fill(False)
            legend_handle.set_linestyle(linestyle)
            legend_handle.set_color(color)
        mplt.xticks([1, 30, 60, 90, 120], ['1', '30', '60', '90', '120'])
        p.save(output)


    def violin(self, df: pd.DataFrame,
            output: str,
            *args,
            **kwargs) -> None:
        p = plt(df, format=["pdf", "png"])
        self.plot(p, "violin", *args, **kwargs)
        p.save(output)

    def boxplot(self, df: pd.DataFrame,
            output: str,
            *args,
            **kwargs) -> None:
        p = plt(df, format=["pdf", "png"])
        self.plot(p, "boxplot", showfliers=False, *args, **kwargs)
        p.save(output)

    def all_window_study(self,
                         data: List[Dict[str, Dict[str, Any]]],
                         labels: List[str]) -> None:
        df = pd.DataFrame(["Label", "Score"])

        for trg, l in zip(data, labels):
            targets: tf.Tensor = trg.dft("Targets")
            tmp_df = pd.DataFrame(
                        {'Label': l,
                         'Score': targets.numpy()}
                    )
            df = pd.concat([df, tmp_df])

        output_file = f"{self.output}/stat_all_scores.pdf"
        colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        cm = sns.color_palette(colors)
        rcParams['figure.figsize'] = 8,4
        rcParams['font.size'] = 25

        self.ecdf(df, output_file,
                  x="Score",
                  hue="Label",
                  hue_order=["U", "B", "G"],
                  palette=cm,
                  linewidth=2.0,
                  set_kwargs={"xlabel": f"Anomalous minutes",
                              "ylabel": "Proportion [%]"},
                  legend_kwargs={"ncol": 3, "x": 0.5, "y": 0.5, "labels": [r"$\hat{G}$", r"$\hat{B}$", r"$\hat{U}$"]})

    def all_window_feature_study(self,
                                 data: List[Dict[str, Dict[str, Any]]],
                                 labels: List[str]) -> None:
        clms = list(data[0]['Windows']['columns'])
        df_clms = ['Label', 'Minute']
        df_clms.extend(clms)
        df = pd.DataFrame(columns=df_clms)

        for trg, l in zip(data, labels):
            targets: tf.Tensor = trg.dft("Windows")
            targets = targets.numpy()
            minute = np.repeat(np.arange(1, targets.shape[1]+1), targets.shape[0])
            targets = np.reshape(targets, (targets.shape[0]*targets.shape[1], targets.shape[2]))
            l_data = {'Label': l, 'Minute': minute}
            trg_data = {clm: targets[:, i] for i, clm in enumerate(clms)}
            l_data.update(trg_data)
            tmp_df = pd.DataFrame(l_data)
            df = pd.concat([df, tmp_df])

        self.write_msg(f"Dataframe: {df}")

        for clm in clms:
            output_file = f"{self.output}/evolution_{clm}_all_windows.pdf"
            colors = ["#e41a1c", "#377eb8", "#4daf4a"]
            cm = sns.color_palette(colors)
            rcParams['figure.figsize'] = 8,4
            rcParams['font.size'] = 25

            self.line(df, output_file,
                      x="Minute",
                      y=clm,
                      hue="Label",
                      hue_order=["U", "B", "G"],
                      palette=cm,
                      linewidth=2.0,
                      set_kwargs={"xlabel": f"Window [mm]",
                                  "ylabel": "Value",
                                  "title": f"{clm} evolution"},
                      legend_kwargs={"ncol": 3, "x": 0.5, "y": -0.5, "labels": [r"$\hat{G}$", r"$\hat{B}$", r"$\hat{U}$"]})


    def window_study(self,
                     data: Dict[str, Dict[str, Any]],
                     label: str) -> None:
        # windows: tf.Tensor = data["Windows"]["tf_values"]
        targets: tf.Tensor = data.dft("Targets")
        expected_label: tf.Tensor = data.dft("ExpectedLabel")
        # origin: tf.Tensor = data["Classes"]["tf_values"]
        # exp_label: tf.Tensor = data["ExpectedLabel"]["tf_values"]

        # wdw_df = self.windows2df(windows)
        # wdw_df["Targets"] = np.repeat(targets.numpy(), windows.shape[1])
        # wdw_df["Origin"] = np.repeat(origin.numpy(), windows.shape[1])
        # wdw_df["Origin"].replace(0, "Good", inplace=True)
        # wdw_df["Origin"].replace(1, "Bad", inplace=True)
        # wdw_df["Origin"].replace(2, "Difficult", inplace=True)
        # wdw_df["Expected_label"] = np.repeat(exp_label.numpy(), windows.shape[1])
        # wdw_df["Expected_label"].replace(0, "Positive", inplace=True)
        # wdw_df["Expected_label"].replace(1, "Negative", inplace=True)

        # self.write_msg(f"------- {label} study --------")
        # threshold = 80.0
        # self.write_msg(f"Targets over {threshold}: {tf.where(targets > threshold)}")

        # print(wdw_df[wdw_df["Expected_label"] == "Positive"]["Targets"].describe())
        # print(wdw_df[wdw_df["Expected_label"] == "Negative"]["Targets"].describe())

        # wdw_df = self.to_long(wdw_df, stubnames="Value", j="Stat")

        # tmp_df = wdw_df[wdw_df["Stat"] == wdw_df["Stat"].unique()[0]]
        output_file = f"{self.output}/stat_{label}-target_exp-label.pdf"
        self.ecdf(targets.numpy(), output_file,
                  set_kwargs={"xlabel": f"VMAF Value",
                              "title": f"{label} dataset VMAF Distribution"},
                  legend_kwargs={"ncol": 2})
        # output_file = f"{self.output}/stat_{label}_origin_stat-comparison.pdf"
        # self.boxplot(wdw_df, output_file, x="Stat", y="Value", hue="Origin",
        #           set_kwargs={"xlabel": "KPI",
        #                       "ylabel": "Value",
        #                       "title": f"{label} dataset statistic distribution"},
        #           legend_kwargs={"ncol": 3})
        # output_file = f"{self.output}/stat_{label}_exp-l_stat-comparison.pdf"
        # self.boxplot(wdw_df, output_file, x="Stat", y="Value", hue="Expected_label",
        #           set_kwargs={"xlabel": "KPI",
        #                       "ylabel": "Value",
        #                       "title": f"{label} dataset statistic distribution"},
        #           legend_kwargs={"ncol": 2})

        # for stat in wdw_df["Stat"].unique():
        #     tmp_df = wdw_df[wdw_df["Stat"] == stat]
        #     output_file = f"{self.output}/stat_{label}-{stat}_origin_ecdf.pdf"
        #     self.ecdf(tmp_df, output_file, x="Value", hue="Origin",
        #               set_kwargs={"xlabel": f"{stat} Value",
        #                           "title": f"{label} dataset {stat} Distribution",
        #                           "xlim": (-10, 10)},
        #               legend_kwargs={"ncol": 3})
        #     output_file = f"{self.output}/stat_{label}-{stat}_exp-l_ecdf.pdf"
        #     self.ecdf(tmp_df, output_file, x="Value", hue="Expected_label",
        #               set_kwargs={"xlabel": f"{stat} Value",
        #                           "title": f"{label} dataset {stat} Distribution",
        #                           "xlim": (-10, 10)},
        #               legend_kwargs={"ncol": 2})

    def window_umap(self,
                    data: Dict[str, Dict[str, Any]],
                    label: str) -> None:
        windows = data["Windows"]["tf_values"].numpy()
        windows = windows.reshape((windows.shape[0], windows.shape[1]*windows.shape[2]))
        targets = data["Targets"]["tf_values"].numpy()
        origin = data["Classes"]["tf_values"].numpy()
        exp_label = data["ExpectedLabel"]["tf_values"].numpy()

        umap_file = f"umap_representation-{label}-dst"
        if self.ph.check(umap_file):
            wdw = self.ph.load(umap_file)
        else:
            wdw = Transform.umap2df(windows)
            self.ph.save(wdw, umap_file)

        wdw["Targets"] = targets
        wdw["Origin"] = origin
        wdw["Origin"].replace(0, "Good", inplace=True)
        wdw["Origin"].replace(1, "Bad", inplace=True)
        wdw["Origin"].replace(2, "Difficult", inplace=True)
        wdw["Expected_Label"] = exp_label
        wdw["Expected_Label"].replace(0, "Positive", inplace=True)
        wdw["Expected_Label"].replace(1, "Negative", inplace=True)

        p = plt(wdw, format=["pdf", "png"])
        p("joint", x="X", y="Y", hue="Origin", kind="kde", fill=True, thresh=0.05, alpha=.8, marginal_kws={'common_norm': False})
        legend_position = (1.18, 1.0)
        # p.set_legend(p.plot.ax_joint, "upper left", bbox_to_anchor=legend_position, ncol=1)
        p.save(f"{self.output}/stat_{label}_umap_origin.pdf")

        p = plt(wdw, format=["pdf", "png"])
        p("joint", x="X", y="Y", hue="Expected_Label", kind="kde", fill=True, thresh=0.05, alpha=.8, marginal_kws={'common_norm': False})
        legend_position = (1.18, 1.0)
        # p.move_legend(p.plot.ax_joint, "upper left", bbox_to_anchor=legend_position, ncol=1)
        p.save(f"{self.output}/stat_{label}_umap_exp_l.pdf")


    def data_dst(self):
        self.write_msg("Studing the datasets distributions")
        # self.write_msg(f"key objects in data:")
        # self.write_msg(f"Training: {self.data.training.keys()}")
        # self.write_msg(f"Validation: {self.data.validation.keys()}")
        # self.write_msg(f"Test: {self.data.test.keys()}")
        # self.write_msg(f"Difficult SL: {self.data.gray_out_train.keys()}")

        # self.window_study(self.data.goods_prop, "Trivial-Positives")
        # self.window_study(self.data.bads_prop, "Trivial-Negatives")
        # self.window_study(self.data.grays_prop, "Difficult")
        self.all_window_study([self.data.goods_prop, self.data.bads_prop, self.data.grays_prop],
                              ["G", "B", "U"])
        self.all_window_feature_study([self.data.goods_prop, self.data.bads_prop, self.data.grays_prop],
                                      ["G", "B", "U"])
        # self.window_study(self.data.training, "Training")
        # self.window_study(self.data.validation, "Validation")
        # self.window_study(self.data.test, "Test")
        # self.window_study(self.data.gray_out_train, "Difficult")

        self.window_umap(self.data.training, "Training")
        self.window_umap(self.data.validation, "Validation")
        self.window_umap(self.data.test, "Test")
        self.window_umap(self.data.gray_out_train, "Difficult")

    def calculate_distances(self,
                            p: np.ndarray,
                            n: np.ndarray,
                            *args,
                            pbar: Optional[tqdm] = None,
                            label: str = "") -> np.ndarray:
            f = f"pos_neg_distances-{label}"

            if self.ph.check(f):
                dst = self.ph.load(f)
                if pbar is not None:
                    pbar.update(p.shape[0]*n.shape[0])
                    pbar.close()
            else:
                comb = np.array(np.meshgrid(np.arange(p.shape[0]), np.arange(n.shape[0]))).T.reshape(p.shape[0]*n.shape[0], -1)

                def distance(elem):
                    # self.write_msg(f"distances between {p[elem[0]], n[elem[1]]}", level=LH.DEBUG)
                    # self.write_msg(f"distances between shapes {p[elem[0]].shape, n[elem[1]].shape}", level=LH.DEBUG)
                    distance, _= fastdtw(p[elem[0]], n[elem[1]], dist=euclidean)
                    if pbar is not None:
                        pbar.update(1)
                    # self.write_msg(f"Distance: {distance}", level=LH.DEBUG)
                    return distance

                v_distances = np.vectorize(distance, signature='(n)->()')
                dst = v_distances(comb)
                dst = tf.convert_to_tensor(dst)
                self.write_msg(f"Distances: {dst.shape}")
                self.ph.save(dst, f)
                pbar.close()
            return dst

    def merge_dimensions(self,
                         new_obj: tf.Tensor,
                         *args,
                         original: Optional[tf.Tensor] = None,
                         axis: int = 1,
                         **kwargs) -> tf.Tensor:
            if original is None:
                res = tf.expand_dims(new_obj, axis)
                self.write_msg(f"Tensor: {res}")
                self.write_msg(f"Tensor shape: {res.shape}")
                return res

            tmp_dst_tf = tf.expand_dims(new_obj, axis)
            self.write_msg(f"tmp Tensor: {tmp_dst_tf}")
            self.write_msg(f"tmp Tensor shape: {tmp_dst_tf.shape}")
            res = tf.concat([original, tmp_dst_tf], axis=axis)
            self.write_msg(f"Tensor: {res}")
            self.write_msg(f"Tensor shape: {res.shape}")
            return res

    def avg_full_dst(self,
                     dst: tf.Tensor,
                     *args,
                     axis: int = 1,
                     **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        avg_wdw_wdw_distances = tf.reduce_mean(dst, axis=axis)
        # self.write_msg(f"Average p_wdw - n_wdw distance Tensor: {avg_wdw_wdw_distances}")
        self.write_msg(f"Average p_wdw - n_wdw distance Tensor shape: {avg_wdw_wdw_distances.shape}")

        avg_pwdw_n_distances = tf.reduce_mean(avg_wdw_wdw_distances, axis=axis)
        # self.write_msg(f"Average p_wdw - n population distance Tensor: {avg_pwdw_n_distances}")
        self.write_msg(f"Average p_wdw - n population distance Tensor shape: {avg_pwdw_n_distances.shape}")
        return avg_wdw_wdw_distances, avg_pwdw_n_distances

    def distance_for_each_dim(self,
                              p: tf.Tensor,
                              n: tf.Tensor,
                              *args,
                              label: str = "-distanceforeachdim-",
                              **kwargs) -> Tuple[tf.Tensor, ...]:
        pos_neg = None
        for dim in range(p.shape[2]):
            pos_dim = p[:, :, dim].numpy()
            neg_dim = n[:, :, dim].numpy()
            self.write_msg(f"positive dimension: {pos_dim}")
            self.write_msg(f"negative dimension: {neg_dim}")
            self.write_msg(f"positive dimension shape: {pos_dim.shape}")
            self.write_msg(f"negative dimension shape: {neg_dim.shape}")

            f_pos_neg = f"{label}-{dim}-"
            pos_neg_dst = self.calculate_distances(pos_dim, neg_dim, label=f_pos_neg)
            self.write_msg(f"all distances: {pos_neg_dst}")
            self.write_msg(f"all distances shape: {pos_neg_dst.shape}")

            pos_neg = self.merge_dimensions(tf.convert_to_tensor(pos_neg_dst), original=pos_neg)

        self.write_msg(f"all dimensions tensor: {pos_neg}")
        self.write_msg(f"all dimensions tensor shape: {pos_neg.shape}")

        pos_wdw_neg_avg, pos_neg_avg = self.avg_full_dst(pos_neg)

        self.write_msg(f"pos_neg_avg: {pos_neg_avg}")
        self.write_msg(f"pos_pos_avg: {pos_pos_avg}")
        self.write_msg(f"neg_neg_avg: {neg_neg_avg}")

        if pos_neg is None:
            raise Exception

        return pos_neg, pos_wdw_neg_avg, pos_neg_avg

    def distance_for_all_dim(self,
                             p: tf.Tensor,
                             n: tf.Tensor,
                             *args,
                             subset_dim: float = 1.0,
                             label: str = "-distanceforeachdim-",
                             **kwargs) -> Tuple[tf.Tensor, ...]:
        pos_neg = None
        pos_dim = p.numpy()
        neg_dim = n.numpy()
        self.write_msg(f"positive dimension: {pos_dim}")
        self.write_msg(f"negative dimension: {neg_dim}")
        self.write_msg(f"positive dimension shape: {pos_dim.shape}")
        self.write_msg(f"negative dimension shape: {neg_dim.shape}")

        if pos_dim.shape[0] == 0 or neg_dim.shape[0] == 0:
            self.write_msg(f"The first dimension of the positives or negatives is 0, nothing to compare")
            return tf.zeros([0]), tf.zeros([0])

        f_pos_neg = f"{label}-allDim-"

        if subset_dim != 1.0:
            n_take = math.ceil(subset_dim*neg_dim.shape[0])
            if n_take == 0:
                raise Exception(f"N_TAKE is 0")
            neg_dim = np.take(neg_dim, np.random.choice(neg_dim.shape[0], n_take), axis=0)

        p_bar = pb.bar(total=pos_dim.shape[0]*neg_dim.shape[0])
        pos_neg_dst = self.calculate_distances(pos_dim, neg_dim, label=f_pos_neg, pbar=p_bar)
        pos_neg_dst = tf.reshape(pos_neg_dst, (p.shape[0], -1))
        self.write_msg(f"all distances: {pos_neg_dst}")
        self.write_msg(f"all distances shape: {pos_neg_dst.shape}")

        avg_dst = tf.reduce_mean(pos_neg_dst, axis=1)
        # self.write_msg(f"Average p_wdw - n_wdw distance Tensor: {avg_dst}")
        self.write_msg(f"Average p_wdw - n_wdw distance Tensor shape: {avg_dst.shape}")

        return pos_neg_dst, avg_dst

    def multi_dataset_similarity(self,
                                 first_dataset: Dict[str, Dict[str, Any]],
                                 second_dataset: Dict[str, Dict[str, Any]],
                                 *args,
                                 first_id: str = "1",
                                 second_id: str = "2",
                                 label: str = "MultiDatasetStudy",
                                 **kwargs) -> None:
        def get_wdw_lbl(data):
            return data["Windows"]["tf_values"], data["ExpectedLabel"]["tf_values"]

        def get_p_n_idx(labels):
            return tf.where(labels == 0)[:, 0], tf.where(labels == 1)[:, 0]

        def get_p_n_wdw(wdw, p_idx, n_idx):
            return tf.gather(wdw, p_idx), tf.gather(wdw, n_idx)

        first_wdw, first_lbl = get_wdw_lbl(first_dataset)
        second_wdw, second_lbl = get_wdw_lbl(second_dataset)

        first_p_idx, first_n_idx = get_p_n_idx(first_lbl)
        second_p_idx, second_n_idx = get_p_n_idx(second_lbl)

        first_p_id, first_n_id = f"{first_id}_p", f"{first_id}_n"
        second_p_id, second_n_id = f"{second_id}_p", f"{second_id}_n"

        first_p_wdw, first_n_wdw = get_p_n_wdw(first_wdw, first_p_idx, first_n_idx)
        second_p_wdw, second_n_wdw = get_p_n_wdw(second_wdw, second_p_idx, second_n_idx)

        f_pos_s_neg, f_pos_s_neg_avg = self.distance_for_all_dim(first_p_wdw, second_n_wdw,
                                                                 label=f"{label}-f_pos_s_neg",
                                                                 **kwargs)
        f_pos_s_neg, f_pos_s_pos_avg = self.distance_for_all_dim(first_p_wdw, second_p_wdw,
                                                                 label=f"{label}-f_pos_s_pos",
                                                                 **kwargs)
        f_neg_s_neg, f_neg_s_neg_avg = self.distance_for_all_dim(first_n_wdw, second_n_wdw,
                                                                 label=f"{label}-f_neg_s_neg",
                                                                 **kwargs)
        f_neg_s_neg, f_neg_s_pos_avg = self.distance_for_all_dim(first_n_wdw, second_p_wdw,
                                                                 label=f"{label}-f_neg_s_pos",
                                                                 **kwargs)

        avg_f_p_s_p_distances_df = pd.DataFrame({"Label": f"{first_p_id}-{second_p_id}", "Value": f_pos_s_pos_avg.numpy()})
        avg_f_p_s_n_distances_df = pd.DataFrame({"Label": f"{first_p_id}-{second_n_id}", "Value": f_pos_s_neg_avg.numpy()})
        avg_f_n_s_p_distances_df = pd.DataFrame({"Label": f"{first_n_id}-{second_p_id}", "Value": f_neg_s_pos_avg.numpy()})
        avg_f_n_s_n_distances_df = pd.DataFrame({"Label": f"{first_n_id}-{second_n_id}", "Value": f_neg_s_neg_avg.numpy()})

        dst_df = pd.concat([avg_f_p_s_p_distances_df,
                            avg_f_p_s_n_distances_df,
                            avg_f_n_s_p_distances_df,
                            avg_f_n_s_n_distances_df])
        dst_df = dst_df.reset_index(drop=True)

        output_file = f"{self.output}/similairty_distances-{label}-ECDF.pdf"
        self.ecdf(dst_df, output_file, x="Value", hue="Label",
                  set_kwargs={"xlabel": f"Distance",
                              "title": f"{label} multi-Dataset inter clusters similarity"},
                  legend_kwargs={"ncol": 3})
        output_file = f"{self.output}/similairty_distances-{label}-BoxPlot.pdf"
        self.boxplot(dst_df, output_file, x="Label", y="Value",
                  set_kwargs={"xlabel": "Similarity Set",
                              "ylabel": "Distance",
                              "title": f"{label} multi-Dataset inter clusters similarity"},
                  legend_kwargs={"ncol": 3})

    def pos_neg_similarity(self,
                           data: Dict[str, Dict[str, Any]],
                           *args,
                           label: str = "PosNegStudy",
                           **kwargs) -> None:
        windows = data["Windows"]["tf_values"]
        expected_label = data["ExpectedLabel"]["tf_values"]

        pos_exp = tf.where(expected_label == 0)[:, 0]
        neg_exp = tf.where(expected_label == 1)[:, 0]

        pos_windows = tf.gather(windows, pos_exp)
        neg_windows = tf.gather(windows, neg_exp)

        self.write_msg(f"Expected positive dimension: {pos_windows.shape[0]}")
        self.write_msg(f"Expected negative dimension: {neg_windows.shape[0]}")

        # self.write_msg(f"First window: {pos_windows[0]}")
        # self.write_msg(f"First window, first dimension: {pos_windows[0, :, 0]}")

        pos_neg, pos_neg_avg = self.distance_for_all_dim(pos_windows, neg_windows,
                                                         label=f"{label}-posneg",
                                                         **kwargs)
        pos_pos, pos_pos_avg = self.distance_for_all_dim(pos_windows, pos_windows,
                                                         label=f"{label}-pospos",
                                                         **kwargs)
        neg_neg, neg_neg_avg = self.distance_for_all_dim(neg_windows, neg_windows,
                                                         label=f"{label}-negneg",
                                                         **kwargs)

        self.write_msg(f"P-N: {pos_neg}")
        self.write_msg(f"P-P: {pos_pos}")
        self.write_msg(f"N-N: {neg_neg}")

        avg_pwdw_n_distances_df = pd.DataFrame({"Label": "P-N", "Value": pos_neg_avg.numpy()})
        avg_pwdw_p_distances_df = pd.DataFrame({"Label": "P-P", "Value": pos_pos_avg.numpy()})
        avg_nwdw_n_distances_df = pd.DataFrame({"Label": "N-N", "Value": neg_neg_avg.numpy()})

        dst_df = pd.concat([avg_pwdw_n_distances_df, avg_pwdw_p_distances_df, avg_nwdw_n_distances_df])
        dst_df = dst_df.reset_index(drop=True)

        output_file = f"{self.output}/similairty_distances-{label}-ECDF.pdf"
        self.ecdf(dst_df, output_file, x="Value", hue="Label",
                  set_kwargs={"xlabel": f"Distance",
                              "title": f"{label} Dataset inter clusters similarity"},
                  legend_kwargs={"ncol": 3})
        output_file = f"{self.output}/similairty_distances-{label}-BoxPlot.pdf"
        self.boxplot(dst_df, output_file, x="Label", y="Value",
                  set_kwargs={"xlabel": "Similarity Set",
                              "ylabel": "Distance",
                              "title": f"{label} Dataset inter clusters similarity"},
                  legend_kwargs={"ncol": 3})


    def similarity_study(self):
        self.write_msg("Studing the Similarity between the time series")
        self.write_msg(f"key objects in data:")
        self.write_msg(f"Training: {self.data.training.keys()}")
        self.write_msg(f"Validation: {self.data.validation.keys()}")
        self.write_msg(f"Test: {self.data.test.keys()}")
        self.write_msg(f"Difficult SL: {self.data.gray_out_train.keys()}")

        # Study the similarity between the expected positive and expected negative
        # population of a dataset
        # self.pos_neg_similarity(self.data.grays_prop, label="Full-Difficult", subset_dim=.5)
        # self.pos_neg_similarity(self.data.goods_prop, label="Full-Goods", subset_dim=.3)
        # self.pos_neg_similarity(self.data.bads_prop, label="Full-bads", subset_dim=.2)
        # self.pos_neg_similarity(self.data.gray_out_train, label="Difficult")
        # self.pos_neg_similarity(self.data.training, label="Training")
        # self.pos_neg_similarity(self.data.validation, label="Validation")
        # self.pos_neg_similarity(self.data.test, label="Test", subset_dim=0.1)

        # Study the similarity between the expected positive and expected negative
        # of a papulation separating also using the problematic applied.
        def sub_data(data_sub_data, idx):
            tmp_data_sub_data = deepcopy(data_sub_data)
            for key in tmp_data_sub_data:
                self.write_msg(f"Gathering property: {key}")
                tmp_data_sub_data[key]["tf_values"] = tf.gather(tmp_data_sub_data[key]["tf_values"], idx)
                self.write_msg(f"{key}: {tmp_data_sub_data[key]['tf_values']}")
            return tmp_data_sub_data

        def sub_dim_data(data_sub_data, dim):
            tmp_data_sub_data = deepcopy(data_sub_data)
            self.write_msg(f"data windows shape: {tmp_data_sub_data['Windows']['tf_values'].shape}")
            tmp_data_sub_data["Windows"]["tf_values"] = tmp_data_sub_data["Windows"]["tf_values"][:, :, dim]
            self.write_msg(f"data windows shape: {tmp_data_sub_data['Windows']['tf_values'].shape}")
            return tmp_data_sub_data

        def study_problem(dst, label, **kwargs):
            for problem in np.unique(dst["Problem"]["tf_values"].numpy()):
                self.write_msg(f"Analyzing problem: {problem}, label: {label}")
                prb_idx = tf.where(dst["Problem"]["tf_values"] == problem)[:, 0]
                self.write_msg(f"Number of idx with such problem: {len(prb_idx)}")
                self.write_msg(f"idx with such problem: {prb_idx}")
                prb_dst = sub_data(dst, prb_idx)
                self.pos_neg_similarity(prb_dst, label=f"{label}-{problem}", **kwargs)

        # study_problem(self.data.gray_out_train, "Difficult", subset_dim=0.5)
        # study_problem(self.data.training, "Training", subset_dim=0.5)
        # study_problem(self.data.validation, "Validation", subset_dim=0.5)
        # study_problem(self.data.test, "Test", subset_dim=0.1)
        # study_problem(self.data.grays_prop, label="Full-Difficult", subset_dim=0.5)
        # study_problem(self.data.goods_prop, label="Full-Goods", subset_dim=0.3)
        # study_problem(self.data.bads_prop, label="Full-bads", subset_dim=0.2)

        def multiDst_study_problem(first_dst, second_dst, label, **kwargs):
            for problem in np.unique(first_dst["Problem"]["tf_values"].numpy()):
                self.write_msg(f"Analyzing problem: {problem}, label: {label}")
                prb_idx = tf.where(first_dst["Problem"]["tf_values"] == problem)[:, 0]
                self.write_msg(f"Number of idx with such problem: {len(prb_idx)}")
                self.write_msg(f"idx with such problem: {prb_idx}")
                prb_dst = sub_data(first_dst, prb_idx)
                self.multi_dataset_similarity(prb_dst, second_dst,
                                              label=f"{label}-{problem}",
                                              **kwargs)

        def multiDst_study_feature(first_dst, second_dst, label: str = "multi-dst-study", **kwargs):
            assert first_dst["Windows"]["tf_values"].shape[2] == second_dst["Windows"]["tf_values"].shape[2]

            for dim in range(first_dst["Windows"]["tf_values"].shape[2]):
                self.write_msg(f"Analyzing dimension {dim}")
                first_sub_dst = sub_dim_data(first_dst, dim)
                second_sub_dst = sub_dim_data(second_dst, dim)
                self.multi_dataset_similarity(first_sub_dst, second_sub_dst,
                                              label=f"{label}-{dim}",
                                              **kwargs)
        self.multi_dataset_similarity(
                    self.data.grays_prop,
                    self.data.training,
                    first_id="D",
                    second_id="T",
                    label="DifficultVSTrivial",
                    subset_dim=0.2
                )
        multiDst_study_feature(
                    self.data.grays_prop,
                    self.data.training,
                    first_id="D",
                    second_id="T",
                    label="DifficultVSTrivial",
                    subset_dim=0.2
                )
        # multiDst_study_problem(
        #             self.data.grays_prop,
        #             self.data.training,
        #             first_id="D",
        #             second_id="T",
        #             label="DifficultVSTrivial",
        #             subset_dim=0.5
        #         )
        # multiDst_study_problem(
        #             self.data.grays_prop,
        #             self.data.goods_prop,
        #             first_id="D",
        #             second_id="G",
        #             label="DifficultVSTrivialGoods",
        #             subset_dim=0.5
        #         )
        # multiDst_study_problem(
        #             self.data.grays_prop,
        #             self.data.bads_prop,
        #             first_id="D",
        #             second_id="B",
        #             label="DifficultVSTrivialBads",
        #             subset_dim=0.5
        #         )


        # Study the similarity using also the other datasets, like difficult
        # vs good ones.

    def feature_correlation(self) -> None:
        data = self.data.data_stats
        features = ['packet_drop_rate',
                    'byte_drop_rate',
                    'avg_timeDelta',
                    'std_timeDelta',
                    'skw_timeDelta',
                    'kur_timeDelta']
        target_feature = 'vmaf'

        p = plt(data, format=['pdf', 'png'])
        for feature in features:
            p("acf", column=feature)
            p.save(f"{self.output}/ACF_{feature}.pdf")

        # Inter feature correlation
        clean_data = data.drop(['second', 'exp_id', 'Dataset'], axis=1)
        clean_data.columns = ['PDR', 'BDR', 'AvgTD', 'StdTD', 'SkwTD', 'KurTD', 'VMAF']
        print(clean_data.corr())
        p = plt(clean_data.corr(), format=['pdf','png'])
        p("heatmap", annot=True, cmap='RdBu', annot_kws={"fontsize":8})
        p.set(title="Feature correlation")
        p.save(f"{self.output}/FeatureCorrelation.pdf")


    def false_positives_ipothesis(self) -> None:
        buoni = self.data.goods_targets.numpy()
        threshold: float = 40.0
        false_positives_idx: List[int] = np.where(buoni < threshold)[0]
        self.logger(self.__class__.__name__, f"Has been detected {len(false_positives_idx)}/{len(buoni)} samples that are classified as good but with a performance lower than the threshold", LH.INFO)
        windows = tf.gather(self.data.goods_windows, false_positives_idx)
        if len(windows) > 0:
            self.logger(self.__class__.__name__, f"Window example: \n{windows[0]}", LH.INFO)
            experiments = []
            for i in range(windows.shape[0]):
                ft = windows[i, 0, :].numpy()
                df = self.data.data
                exp = df[(df["packet_drop_rate"] == ft[0]) & \
                         (df["byte_drop_rate"] == ft[1]) & \
                         (df["avg_timeDelta"] == ft[2]) & \
                         (df["std_timeDelta"] == ft[3]) & \
                         (df["skw_timeDelta"] == ft[4]) & \
                         (df["kur_timeDelta"] == ft[5])]["exp_id"].values
                j = 1
                while len(exp) > 1:
                    ft = windows[i, j, :].numpy()
                    j += 1
                    tmp_exp = df[(df["packet_drop_rate"] == ft[0]) & \
                                 (df["byte_drop_rate"] == ft[1]) & \
                                 (df["avg_timeDelta"] == ft[2]) & \
                                 (df["std_timeDelta"] == ft[3]) & \
                                 (df["skw_timeDelta"] == ft[4]) & \
                                 (df["kur_timeDelta"] == ft[5])]["exp_id"].values
                    exp = np.intersect1d(exp, tmp_exp)
                experiments.append(exp)
            self.logger(self.__class__.__name__, f"Experiments ids: \n{experiments}", LH.INFO)

    def target_distribution_study(self) -> None:
        buoni_trg = self.data.goods_targets.numpy()
        cattivi_trg = self.data.bads_targets.numpy()
        grigi_trg = self.data.grays_targets.numpy()

        buoni = pd.DataFrame({'Dataset': "Good", 'Stat': "VMAF", 'Value': buoni_trg})
        cattivi = pd.DataFrame({'Dataset': "Bad", 'Stat': "VMAF", 'Value': cattivi_trg})
        grigi = pd.DataFrame({'Dataset': "Gray", 'Stat': "VMAF", 'Value': grigi_trg})

        full_df = pd.concat([buoni, cattivi, grigi])
        full_df.reset_index(drop=True, inplace=True)

        self.statPDF(full_df,
                     self.params["studyOutput"],
                     appendix="_VMAFDist")
        self.statECDF(full_df,
                     self.params["studyOutput"],
                     appendix="_VMAFDist")

    def average_windows_study(self) -> None:
        buoni = self.data.goods_windows
        cattivi = self.data.bads_windows
        grigi = self.data.grays_windows

        buoni = tf.math.reduce_max(buoni, 1).numpy()
        cattivi = tf.math.reduce_max(cattivi, 1).numpy()
        grigi = tf.math.reduce_max(grigi, 1).numpy()

        buoni = pd.DataFrame(buoni, columns=self.stat_columns)
        cattivi = pd.DataFrame(cattivi, columns=self.stat_columns)
        grigi = pd.DataFrame(grigi, columns=self.stat_columns)

        def to_long(df):
            df["i"] = np.arange(len(df.index))
            df = pd.wide_to_long(df, stubnames="Value", i=["i"], j="Stat", suffix=".*").reset_index()
            df = self.actions.get_handler("dropColumns")(
                    df = df,
                    columns = ["i"]
                 )
            return df
        buoni = to_long(buoni)
        cattivi = to_long(cattivi)
        grigi = to_long(grigi)

        buoni["Dataset"] = "Buoni"
        cattivi["Dataset"] = "Cattivi"
        grigi["Dataset"] = "Grigi"

        full_df = pd.concat([buoni, cattivi, grigi])
        full_df.reset_index(drop=True, inplace=True)
        self.statPDF(full_df,
                     self.params["studyOutput"],
                     appendix="_windowMax")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.actions.get_handler("dropColumns")(
                df = df,
                columns = ["vmaf"]
             )
        df.columns = ["Second",
                      "ValuePDR",
                      "ValueBDR",
                      "ValueAvgIPTD",
                      "ValueStdIPTD",
                      "ValueSkwIPTD",
                      "ValueKurIPTD",
                      "exp_id"]
        df = pd.wide_to_long(df, stubnames="Value", i=['exp_id', "Second"], j="Stat", suffix=".*")
        df.reset_index(inplace=True)
        df = self.actions.get_handler("dropColumns")(
                df = df,
                columns = ["Second", "exp_id"]
             )
        return df

    def statPDF(self, df: pd.DataFrame,
                output: str,
                appendix: str = "") -> None:
        for stat in df["Stat"].unique():
            fileName = f"{output}/stat_{stat}_pdf{appendix}{self.hash}.pdf"
            if not FH.exists(fileName):
                stat_df = df[df["Stat"] == stat]
                p = plt(stat_df, format=["pdf", "png"])
                p.pdf(x="Value", hue="Dataset", cumulative=True, common_norm=False, common_grid=True)
                p.set(title=f"{stat} distribution among datasets",
                      xlim=self.stat_xlims[stat])
                p.save(fileName)

    def statECDF(self, df: pd.DataFrame,
                 output: str,
                 appendix: str = "") -> None:
        for stat in df["Stat"].unique():
            fileName = f"{output}/stat_{stat}_ecdf{appendix}{self.hash}.pdf"
            if not FH.exists(fileName):
                stat_df = df[df["Stat"] == stat]
                p = plt(stat_df, format=["pdf", "png"])
                p.ecdf(x="Value", hue="Dataset")
                p.set(title=f"{stat} distribution among datasets",
                      xlim=self.stat_xlims[stat])
                p.save(fileName)
