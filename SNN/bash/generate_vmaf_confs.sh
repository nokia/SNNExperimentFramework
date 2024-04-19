#!/bin/bash

LC_NUMERIC=C
base_folder="Conf/Infocomm/SNN/VMAF"
base_file="base.yaml"
base_file_path="${base_folder}/${base_file}"

vmaf_thresholds=$(seq 10 10 90)
rngs=$(seq 1 1 10)

rm -rf ${base_folder}/VMAFConf-*.yaml

for vmaf in ${vmaf_thresholds[*]}
do
		for rng in ${rngs[*]}
		do
				new_file="VMAFConf-${vmaf}-${rng}.yaml"
				new_file_path="${base_folder}/${new_file}"
				cp ${base_file_path} ${new_file_path}
				sed -i "s|\$threshold|${vmaf}|g" ${new_file_path}
				sed -i "s|\$rng|${rng}|g" ${new_file_path}
		done
done

for file in ${base_folder}/VMAFConf-*.yaml
do
		# snn -c ${file} -t --csv ${base_folder}/statistics.csv
		echo "test"
done
