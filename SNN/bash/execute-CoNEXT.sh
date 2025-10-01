#!/bin/bash

LC_NUMERIC=C

main_folder="Conf/CoNEXT"
s=13
f=13

#for e in Trivial SL; do
#        input="${main_folder}/${e}"
#        for d in Bottle Bunny ScarletV3; do
#                for v in 80 90 99; do
#                        sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
#                        joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
#                        echo ${sub_exp}
#                        parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study " ::: $(seq ${s} ${f});
#                done
#        done
#done

for e in SL-RL; do
		input="${main_folder}/${e}/"
		#for d in Bottle; do
		#        for v in 80; do
		#                for k in {7..14}; do
		#                        sub_exp="${input}/${d}-VMAF${v}-multiExp-${k}/${d}-VMAF${v}-"
		#                        joblog="${input}/${d}-VMAF${v}-multiExp-${k}/JobLog-Bottle-${k}.parallel"
		#                        echo "${e}-${d}-${v}"
		#                        parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: 13 18 19 22 26 29 31 36 38 48
		#                done
		#        done
		#done
		#for d in Bottle; do
		#        for v in 80 90; do
		#                for k in 2; do
		#                        sub_exp="${input}/${d}-VMAF${v}-multiExp-${k}/${d}-VMAF${v}-"
		#                        joblog="${input}/${d}-VMAF${v}-multiExp-${k}/JobLog-Bottle-${k}-part3.parallel"
		#                        echo "${e}-${d}-${v}"
		#                        parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 80 99);
		#                done
		#        done
		#done
		for d in Bunny; do
				for v in 99; do
						sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
						joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-${d}-part4.parallel"
						echo "${e}-${d}-${v}"
						parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 60 80);
				done
		done
		for d in ScarletV3; do
				for v in 90 99; do
						sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
						joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-${d}-part4.parallel"
						echo "${e}-${d}-${v}"
						parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 60 80);
				done
		done
		#for d in ScarletV3; do
		#        for v in 80; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-Scarlet-1.parallel"
		#                echo "${e}-${d}-${v}"
		#                parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 10 18);
		#        done
		#        for v in 90; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                joblog="${input}/${d}-VMAF${v}-multiExp/JobLog-Scarlet-2.parallel"
		#                echo "${e}-${d}-${v}"
		#                parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 50 50);
		#        done
		#done
		#for d in Bottle; do
		#        for v in 80; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                #joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                #echo ${e}-${d}-${v}
		#                ##parallel -j 1 --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 0 0);
		#                #parallel -j 1 --joblog ${joblog}-2 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 2 9);
		#                parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 1 1);
		#        done
		#        for v in 90; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                #joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                #echo ${e}-${d}-${v}
		#                ##parallel -j 1 --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 0 0);
		#                #parallel -j 1 --joblog ${joblog}-2 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 2 9);
		#                #parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 2 2);
		#                parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 0 0);
		#                parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 9 9);
		#        done
		#done
		#for d in ScarletV3; do
		#        #for v in 80; do
		#        #        sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#        #        echo ${e}-${d}-${v}
		#        #        #parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 3 3);
		#        #done
		#        #for v in 90; do
		#        #        sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#        #        echo ${e}-${d}-${v}
		#        #        parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 4 4);
		#        #done
		#        for v in 99; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                echo ${e}-${d}-${v}
		#                parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 0 9);
		#        done
		#done
		#for d in Bunny ; do
		#        for v in 80; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 5 9);
		#        done
		#        for v in 90 99; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                echo ${e}-${d}-${v}
		#                parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 0 9);
		#        done
		#done
		#for d in Football; do
		#        for v in 80; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                #joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                echo ${e}-${d}-${v}
		#                parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 2 3);
		#                parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 6 9);
		#        done
		#done
		#for d in Football; do
		#        for v in 90 99; do
		#                sub_exp="${input}/${d}-VMAF${v}-multiExp/${d}-VMAF${v}-"
		#                joblog="${input}/${d}-VMAF${v}-multiExp/JobLog.parallel"
		#                echo ${e}-${d}-${v}
		#                parallel -j 1 --resume --joblog ${joblog} --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 0 9);
		#        done
		#done
done
