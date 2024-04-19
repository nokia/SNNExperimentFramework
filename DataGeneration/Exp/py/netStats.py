import os
import math
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import more_itertools as mit

columns = ["frame", "arrival_time", "diff_time", "frame_len"]

parser = argparse.ArgumentParser(usage="usage: netStats.py [options]",
                                 description="Use the script to generate the statistics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", dest="csv_file", default="result/tshark.csv",
                    action="store", help="define the csv input file")
parser.add_argument("-i", "--id", dest="id", default=0, action="store",
                    help="define the experiment id")
parser.add_argument("-o", "--output", dest="output", default="stats.csv",
                    action="store", help="Define the file where to store the stats")
parser.add_argument("-v", "--vmaf", dest="vmaf", default=100,
                    action="store", help="Define the vmaf value to associate")
parser.add_argument("-p", "--problem", dest="problem", default="None",
                    action="store", help="Define the problem that caused those statistics")
parser.add_argument("-P", dest="p_flag", default=True, action="store_false",
                    help="Disable the problem column")
parser.add_argument("-r", "--reverse", dest="reverse", default=False,
                    action="store_true", help="activate the reverse option, by default the file will be read backword, the second 1 is the last second of packets received")


def get_duration(df: pd.DataFrame) -> float:
    start = df[columns[1]].values[0]
    end = df[columns[1]].values[-1]
    return round(end-start, 3)

def get_length(df: pd.DataFrame) -> int:
    return len(df.index)

def calculate_pps(df: pd.DataFrame) -> float:
    duration = get_duration(df)
    num_of_frames = get_length(df)
    return num_of_frames/duration

def calculate_packet_size(df: pd.DataFrame) -> float:
    return df[columns[3]].mean()

def calculate_all_bytes(df: pd.DataFrame) -> float:
    return df[columns[3]].sum()

def calculate_bps(df: pd.DataFrame) -> float:
    return calculate_all_bytes(df)/get_duration(df)

def calculate_avg_td(df: pd.DataFrame) -> float:
    return df[columns[2]].mean()

def calculate_std_td(df: pd.DataFrame) -> float:
    return df[columns[2]].std()

def calculate_skew_td(df: pd.DataFrame) -> float:
    return df[columns[2]].skew()

def calculate_kurtosis_td(df: pd.DataFrame) -> float:
    return df[columns[2]].kurtosis()

def calculate_outliers_td(df: pd.DataFrame) -> float:
    Q1 = df[columns[2]].quantile(0.05)
    Q3 = df[columns[2]].quantile(0.95)
    IQR = Q3 - Q1
    lower_lim = Q1 - (1.5*IQR)
    upper_lim = Q3 + (1.5*IQR)
    lower_points = sum(df[columns[2]].values < lower_lim)
    upper_points = sum(df[columns[2]].values > upper_lim)
    return lower_points + upper_points

def calculate_id_order(df: pd.DataFrame) -> float:
    ids = list(df[columns[4]].values)
    ids_ordered = sorted(ids)
    tau, p_val = stats.kendalltau(ids, ids_ordered)
    if tau > 0.9999:
        tau = 1.0
    return tau

def count_disordered(l, window = 16) -> int:
    actual_list = range(0, window)
    dist = 0

    ids = list(mit.locate(l, lambda x: x == 0))
    while len(ids) > 1:
        tmp = l[ids[0]:ids[1]]
        if len(tmp) == len(actual_list):
            t, _ = stats.kendalltau(tmp, actual_list)
        else:
            t = 0

        if t < 1:
            dist += 1

        ids.pop(0)

    return dist

def calculate_mp2t_disorder(df: pd.DataFrame) -> float:
    pids = df[columns[5:12]]
    values = df[columns[12:19]]
    limit = 15

    obj = {}
    for p, v in zip(pids.values, values.values):
        mp2t = zip(p, v)
        for elem in mp2t:
            if elem[0] not in obj.keys():
                obj[elem[0]] = [elem[1]]
            else:
                obj[elem[0]].append(elem[1])

    distor = {}
    for key in obj:
        if type(obj[key][0]) == np.float64:
            print(df)
            print(f"obj: {obj[key][0]}, key: {key}")
        tmp = list(range(0, obj[key][0]))
        tmp.extend(obj[key])
        tmp.extend(list(range(obj[key][-1]+1, limit+1)))
        distor[f"mp2t_disorder_{key}"] = count_disordered(tmp, window=limit+1)

    return distor

def get_df_net_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    stats["packets"] = get_length(df)
    stats["bytes"] = calculate_all_bytes(df)
    stats["avg_timeDelta"] = calculate_avg_td(df)
    stats["std_timeDelta"] = calculate_std_td(df)
    stats["skw_timeDelta"] = calculate_skew_td(df)
    stats["kur_timeDelta"] = calculate_kurtosis_td(df)
    #stats["out_timeDelta"] = calculate_outliers_td(df)
    # stats["pkts_order"] = calculate_id_order(df)
    #stats["mp2t_order"] = calculate_mp2t_order(df)
    # stats.update(calculate_mp2t_disorder(df))
    return stats

def get_window_df(df: pd.DataFrame, window) -> pd.DataFrame:
    duration = get_duration(df)
    delta = duration - window
    delta = round(delta, 3)
    return df[df[columns[1]] >= delta]

def main():
    options = parser.parse_args()
    input: str = options.csv_file
    output: str = options.output
    vmaf: float = options.vmaf
    problem: str = options.problem
    p_flag: str = options.p_flag
    id: int = options.id
    time_span = 1

    if os.path.isfile(output):
        resulting_df = pd.read_csv(output)
    else:
        resulting_df = pd.DataFrame()

    if vmaf == "0.0":
        print("Vmaf is 0, no packets to evaluate")
        sys.exit(1)

    df = pd.read_csv(input)
    stats = {}
    single_df = pd.DataFrame()
    for i in range(1, math.ceil(get_duration(df))+1):
        tmp_df = df[(df[columns[1]] < i) & (df[columns[1]] >= i-1)]

        stats.update(get_df_net_stats(tmp_df))
        stats["second"] = i
        single_df = single_df.append(stats, ignore_index=True)

    single_df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
