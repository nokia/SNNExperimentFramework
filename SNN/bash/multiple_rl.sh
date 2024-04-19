#!/bin/bash

LC_NUMERIC=C
i_start=0.3
i_step=0.1
i_end=0.5
d_start=1
d_step=1
d_end=1

#snn -c Conf/H264/RL/SNNNoPenalty.yaml  -t
#snn -c Conf/H264/RL/ACBk.yaml  -t
base_folder="Conf/H264/RL/"
folder="penetration/11-07/"

#for d in $(seq ${d_start} ${d_step} ${d_end})
#do
for i in $(seq ${i_start} ${i_step} ${i_end})
do
		echo "${i}"
		snn -c ${base_folder}${folder}/val-${i}.yaml -t -vvvvvv
done
#done

#snn -c Conf/H264/RL/FPplus.yaml  -t
#snn -c Conf/H264/RL/FNplus.yaml  -t
#snn -c Conf/H264/RL/K-val/multiExp/d1/val-0.1.yaml -t
#snn -c Conf/H264/RL/5actions.yaml  -t
