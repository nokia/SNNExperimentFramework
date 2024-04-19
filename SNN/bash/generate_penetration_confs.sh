#!/bin/bash

LC_NUMERIC=C
base_folder="Conf/Infocomm/SNN/Penetration"
base_file="base.yaml"
base_file_path="${base_folder}/${base_file}"

p_penetration=$(seq 0.1 0.1 1.0)
n_penetration=$(seq 0.9 0.1 0.)
rngs=$(seq 1 1 10)

rm -rf ${base_folder}/PenetrationConf-*.yaml

for vmaf in ${vmaf_thresholds[*]}
do
		for rng in ${rngs[*]}
		do
				new_file="PenetrationConf-${vmaf}-${rng}.yaml"
				new_file_path="${base_folder}/${new_file}"
				cp ${base_file_path} ${new_file_path}
				sed -i "s|\$threshold|${vmaf}|g" ${new_file_path}
				sed -i "s|\$rng|${rng}|g" ${new_file_path}
		done
done

for file in ${base_folder}/PenetrationConf-*.yaml
do
		# snn -c ${file} -t --csv ${base_folder}/statistics.csv
		echo "test"
done
