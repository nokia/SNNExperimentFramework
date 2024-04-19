#!/bin/bash

acce01="Conf/NFV-SDN/multi-exp/categorical-500-acce-01-i1/*"
baseline="Conf/NFV-SDN/multi-exp/baseline_tmp/*"
baseline_100="Conf/NFV-SDN/multi-exp/baseline_100_tmp/*"
acce_trick="Conf/NFV-SDN/multi-exp/categorical-500-acce-1-i10/*"
acce="Conf/NFV-SDN/multi-exp/categorical-500-acce-1-i10_tmp/*"
acce_triangular="Conf/NFV-SDN/multi-exp/categorical-500-acce-1-i10-noTriangular/*"
hash=('1126cc3ee9732526' '30e9d46a48246c81' 'abc8fa67ebb910c3' '344c0037c588991b' 'ac77f6beeb03edaa' '205a26c440290b0c' 'bcd201c2271a8b24' '00e4fdba4fe51db1' '6b54ff25283869a4' '8682fd72e9ac1664')
hash_acce=('bcd201c2271a8b24' '6b54ff25283869a4')

parallel -j 1 --bar snn -c {1} -H {2} --study ::: ${acce01} :::+ ${hash[@]}
exit 0
#parallel -j 1 --bar snn -c {1} -H {2} --study ::: ${baseline_100} :::+ ${hash[@]}
#parallel -j 1 --bar snn -c {1} -H {2} --study ::: ${acce} :::+ ${hash[@]}
#parallel -j 1 --bar snn -c {1} -H {2} --study ::: ${acce_trick} :::+ ${hash[@]}
#parallel -j 1 snn -c {1} -H {2} --study ::: ${acce_triangular} :::+ ${hash[@]}
#parallel -j 1 echo {1} {2} ::: ${acce} :::+ ${hash[@]}
#parallel -j 1 echo {1} {2} ::: ${acce_triangular} :::+ ${hash[@]}
# parallel -j 1 "echo -c {} -H {} --study" ::: ${confs} ::: ${hash};
# parallel -j 1 --bar "snn -c ${sub_exp}{} --study --reinforcement" ::: $(seq 1 1);
