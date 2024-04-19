import seaborn as sns
import pandas as pd
from SNN2.src.io.progressBar import pb

d = "results-bunny-13-04-2023/SL-OTTRL-multiExp/exp-20"
f = f"{d}/csv/reinforcement_step_test.csv"

df = pd.read_csv(f)

bar = pb.bar(len(df["Episode"].unique()))
for episode in df["Episode"].unique():
    tmp_df = df[df["Episode"] == episode]
    steps = tmp_df["Step"].unique()
    print(tmp_df["Statistic"].unique())
    stats = tmp_df[tmp_df["Statistic"].isin(["ActualReward", "PreviousReward"])
bar.close()

