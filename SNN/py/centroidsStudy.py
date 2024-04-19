import os
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.plot.plotter import plotter as plt

from typing import Dict, List, Any, Callable, Tuple

parser = argparse.ArgumentParser(usage="usage: centroidStudy.py [options]",
                                 description="Use the script to study samples vs centroids",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--pfile", dest="positive_embeddings", default=None,
                    action="store", help="define the positive embeddings file")
parser.add_argument("-a", "--afile", dest="anchor_embeddings", default=None,
                    action="store", help="define the anchor embeddings file")
parser.add_argument("-n", "--nfile", dest="negative_embeddings", default=None,
                    action="store", help="define the negative embeddings file")
parser.add_argument("-c", "--classes", dest="classes_file", default=None,
                    action="store", help="define the classes file")
parser.add_argument("-l", "--label", dest="label", default="0.5M",
                    action="store", help="define the label")
parser.add_argument("-o", "--output", dest="output_folder", default="plots/",
                    action="store", help="Define the output folder where to store the plots")
parser.add_argument("-x", "--appendix", dest="appendix", default="iteration0",
                    action="store", help="Define the appendix")
parser.add_argument("--append", dest="append", default=True,
                    action="store_false", help="Disable the append method")

def load(file: str) -> Any:
    with open(file, 'rb') as f:
        return pkl.load(f)

def save(file: str, obj: Any) -> None:
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists")
    with open(file, 'wb') as f:
        pkl.dump(obj, f)

def calculate_centroid(embeddings: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(embeddings, 0)

def calculate_distances(samples: tf.Tensor, centroid: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.square(samples - centroid), -1)

def separate_classes(embeddings: tf.Tensor, classes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.gather(embeddings, tf.where(classes==0)), \
           tf.gather(embeddings, tf.where(classes==1))

def concatenate(objs: List[tf.Tensor]) -> tf.Tensor:
    return tf.concat(objs, 0)

def df_save(df: pd.DataFrame, output: str) -> None:
    if FH.exists(output):
        df.to_csv(output, index=False, header=False, mode='a')
    else:
        df.to_csv(output, index=False)


def main():
    options = parser.parse_args()
    embp_fh: FH = FH(options.positive_embeddings, create=False)
    emba_fh: FH = FH(options.anchor_embeddings, create=False)
    embn_fh: FH = FH(options.negative_embeddings, create=False)
    classes: FH = FH(options.classes_file, create=False)
    label: List[str] = options.label
    output: str = options.output_folder
    appendix: str = options.appendix

    embp = load(embp_fh.path)
    emba = load(emba_fh.path)
    embn = load(embn_fh.path)
    cls = load(classes.path)

    assert embp.shape[0] == emba.shape[0]
    assert embp.shape[0] == embn.shape[0]
    assert embp.shape[0] == cls.shape[0]

    embp_p, embp_n = separate_classes(embp, cls)
    embn_n, embn_p = separate_classes(embn, cls)
    emba_p, emba_n = separate_classes(emba, cls)

    embp = concatenate([embp_p, embn_p])
    embn = concatenate([embp_n, embn_n])

    embp_centroid = calculate_centroid(embp)
    embn_centroid = calculate_centroid(embn)

    emba_p_pcentroid_distances = calculate_distances(emba_p, embp_centroid).numpy()
    emba_p_ncentroid_distances = calculate_distances(emba_p, embn_centroid).numpy()
    emba_n_pcentroid_distances = calculate_distances(emba_n, embp_centroid).numpy()
    emba_n_ncentroid_distances = calculate_distances(emba_n, embn_centroid).numpy()
    centroid_distance = calculate_distances(embp_centroid, embn_centroid)

    assert len(emba_p_pcentroid_distances) == len(emba_p_ncentroid_distances)
    assert len(emba_n_pcentroid_distances) == len(emba_n_ncentroid_distances)

    classes = np.concatenate((
                    np.repeat(["P", "P"], len(emba_p_pcentroid_distances)),
                    np.repeat(["N", "N"], len(emba_n_pcentroid_distances))
                ))
    distances = np.repeat([["AP", "AN"]], len(emba_p_pcentroid_distances)+len(emba_n_pcentroid_distances), axis=0).flatten()
    emba_p_dst = np.stack((emba_p_pcentroid_distances, emba_p_ncentroid_distances), axis=-1).flatten()
    emba_n_dst = np.stack((emba_n_pcentroid_distances, emba_n_ncentroid_distances), axis=-1).flatten()
    emba_dst = np.concatenate((emba_p_dst, emba_n_dst))

    df = pd.DataFrame({
            "label": label,
            "Class": classes,
            "Distances": distances,
            "Value": emba_dst
        })

    output_dst_file = f"{output}/sample_vs_centroid_distances{appendix}.csv"
    df_save(df, output_dst_file)

    centr_dst = pd.DataFrame({
            "label": label,
            "InterCentroidDST": centroid_distance.numpy()
        })

    output_dst_file = f"{output}/inter_centroid_distance{appendix}.csv"
    df_save(centr_dst, output_dst_file)

if __name__ == "__main__":
    main()
