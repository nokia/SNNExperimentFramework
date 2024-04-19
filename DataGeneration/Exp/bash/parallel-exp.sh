#!/bin/bash

LC_NUMERIC=C

videos_delay=("bottle" "football" "scarlet")
videos_drop=("bottle" "football" "scarlet")
videos_good=("bottle" "football" "scarlet")
delay_seq=$(seq 20 10 90)
drop_seq=$(seq 0.05 0.05 3)
problem_reps=10
good_reps=0
parallel_jobs=1
joblog_delay=$(pwd)/test-parallel-DELAY/log-last-experiments-scarlet
joblog=$(pwd)/test-parallel/log-last-experiments-good

execute() {
		#echo "${1} - ${2} - ${3} - ${4}"
		if [ $2 == "good" ]; then
				#echo "good: dkr5g -d Parallel/${1}-${2} 2>/dev/null 1>&2"
				dkr5g -d test-parallel/${1}-${2} 1>/dev/null
		else
				#echo "not good: dkr5g -d Parallel/${1}-${2}-${3} 2>/dev/null 1>&2"
				#dkr5g -d Parallel/${1}-${2}-${3} 2>/dev/null 1>&2
				dkr5g -d test-parallel/${1}-${2}-${3} 1>/dev/null
		fi
}
export -f execute

parallel -j ${parallel_jobs} --resume --joblog ${joblog_delay} --bar execute ::: ${videos_delay[*]} ::: delay ::: ${delay_seq} ::: $(seq ${problem_reps})
parallel -j ${parallel_jobs} --resume --joblog ${joblog} --bar execute ::: ${videos_drop[*]} ::: drop ::: ${drop_seq} ::: $(seq ${problem_reps})
parallel -j ${parallel_jobs} --joblog ${joblog} --bar execute ::: ${videos_good[*]} ::: good ::: $(seq ${good_reps})
