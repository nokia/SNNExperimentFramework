#!/bin/bash

LC_NUMERIC=C

main_folder="Conf/CoNEXT"
s=0
f=50

for e in Trivial SL; do
		input="${main_folder}/${e}"
		for d in Bottle Bunny Scarlet; do
				for v in 80 90 99; do
						sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
						joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
						echo ${sub_exp}
						parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study " ::: $(seq ${s} ${f});
				done
		done
done

for e in SL-RL; do
		input="${main_folder}/${e}/"
		for d in Bottle; do
				for v in 80 90 99; do
					sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
					joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-Bottle-${k}.parallel"
					echo "${e}-${d}-${v}"
					parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq ${s} ${f});
				done
		done
		for d in Bunny; do
				for v in 80 90 99; do
						sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
						joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-${d}-part4.parallel"
						echo "${e}-${d}-${v}"
						parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq ${s} ${f});
				done
		done
		for d in Scarlet; do
				for v in 80 90 99; do
						sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
						joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-${d}-part4.parallel"
						echo "${e}-${d}-${v}"
						parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq ${s} ${f});
				done
		done
done
