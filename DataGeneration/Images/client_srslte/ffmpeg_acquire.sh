#!/bin/bash

frames=30
sdp="sdp_file.sdp"
video_output="local.mp4"
output="log"
csv_file="metrics.csv"

while getopts "f:s:v:o:c:" OPTNAME
do
	case $OPTNAME in
		f) frames=$OPTARG;;
		s) sdp=$OPTARG;;
		v) video_output=$OPTARG;;
		o) output=$OPTARG;;
		c) csv_file=$OPTARG;;
	esac
done

su vlcuser -c "echo \"frame,arrival_time,diff_time,frame_len,id,PID1,PID2,PID3,PID4,PID5,PID6,PID7,cc1,cc2,cc3,cc4,cc5,cc6,cc7\" > ${csv_file}"
su vlcuser -c "tshark -i tun_srsue -q -T fields -E separator=, -e frame.number -e frame.time_relative -e frame.time_delta -e frame.len -e ip.id -e mp2t.pid -e mp2t.cc >> ${csv_file} &"

# -max_muxing_queue_size 5000 \
# -rtbufsize 150M
# -probesize 32 -analyzeduration 0 \
# -fps_mode[:stream_specifier] parameter
# -buffer_size

ffmpeg -buffer_size 3000 -nostdin -use_wallclock_as_timestamps 1 \
		-protocol_whitelist file,rtp,udp \
		-i ${sdp} \
		-fps_mode cfr \
		-framerate ${frames} -c:v copy \
		-flags +global_header -movflags faststart \
		${video_output} -y 2>&1 | tee ${output}_ffmpeg.out
