#!/bin/bash

# tc part


dst="tmp/result/res.mp4";
stream="rtp://server:8554/test";
loss=0.40
log_file="tmp/result/logfile.pcap"
timeout=60
header=200
port=8554
extension="mpg"
netCaching=2000
output="/tmp/result/"

while getopts "s:f:t:l:h:p:e:n:r:" OPTNAME
do
   case $OPTNAME in
		s) stream=$OPTARG;;
		f) dst=$OPTARG;;
	    t) timeout=$OPTARG;;
	    l) log_file=$OPTARG;;
		h) header=$OPTARG;;
		p) port=$OPTARG;;
		e) extension=$OPTARG;;
		n) netCaching=$OPTARG;;
		r) output=$OPTARG;;
	    *) echo "Unrecognized option"
   esac
done

#su vlcuser -c "tcpdump -i tun_srsue -s ${header}  -w "$log_file" udp dst port "$port" and inbound &"

su vlcuser -c "echo \"frame,arrival_time,diff_time,frame_len,id,PID1,PID2,PID3,PID4,PID5,PID6,PID7,cc1,cc2,cc3,cc4,cc5,cc6,cc7\" > ${log_file}"
#su vlcuser -c "tshark -i tun_srsue -q -T fields -E separator=, -e frame.number -e frame.time_relative -e frame.time_delta -e frame.len -e ip.id -e mp2t.pid -e mp2t.cc >> ${log_file} &"
su vlcuser -c "tshark -i tun_srsue -w ${log_file}.pcap &"

#su vlcuser -c "timeout "$timeout"s cvlc -vvv --network-caching 100 $stream --sout=file/mp4:${dst}"
#su vlcuser -c "timeout "$timeout"s cvlc -vvv --network-caching ${netCaching} $stream --sout=file/${extension}:${dst}"
su vlcuser -c "timeout "$timeout"s cvlc -vvv $stream --network-caching ${netCaching} --sout=file/${extension}:${dst}" 2>&1 | tee ${output}vlc.out

pkill tshark

