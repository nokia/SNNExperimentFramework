#!/bin/bash

interface=tun_srsenb
output=/tmp/capture.pcap
while getopts "i:w:" OPTNAME
do
		case $OPTNAME in
				i) interface=$OPTARG;;
				w) output=$OPTARG;;
		esac
done

su vlcuser -c "tshark -i ${interface} -w ${output} &"
