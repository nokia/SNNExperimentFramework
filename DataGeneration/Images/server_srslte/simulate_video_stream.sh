#!/bin/bash

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P );
name=$SCRIPTPATH"/counter.mp4";
sdp="client:8554";
local="local.mp4"
caching="2000"
output="/tmp/result/"

while getopts "f:i:o:l:r:" OPTNAME
do
   case $OPTNAME in
		f) name=$OPTARG;;
		i) sdp=$OPTARG;;
		o) local=$OPTARG;;
		l) caching=$OPTARG;;
		r) output=$OPTARG;;
   esac
done

#sout="#transcode{vcodec=h264,vb=800,scale=0,acodec=mpga,ab=128,channels=2,samplerate=44100}:duplicate{dst=udp{mux=ts,dst="$sdp"},dst=file{dst=/tmp/result/"$local"}}"
#sout="#transcode{vcodec=mp2v,fps=24}:duplicate{dst=udp{mux=ts,dst="$sdp"},dst=file{dst=/tmp/result/"$local"}}"
sout="#duplicate{dst=udp{mux=ts,dst=${sdp}},dst=file{dst=${output}${local}}}"
#su vlcuser -c "cvlc -vvv --play-and-exit --live-caching ${caching} $name --sout '$sout'"
echo ${sout}
su vlcuser -c "cvlc -vvv --play-and-exit ${name} --sout '${sout}'" 2>&1 | tee ${output}vlc.out

