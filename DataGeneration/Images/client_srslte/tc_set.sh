#!/bin/bash

eth=eth0
delay=0ms
loss=0%
rate="100mbit burst 1600 limit 3000"
wait=0

PARSED_ARGUMENTS=$(getopt -a -n experiments -o d:i:l:r:w: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
		echo "Invalid arguments"
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
	-d)   			  delay="$2"      	; shift  2 ;;
    -i)   			  eth="$2"      	; shift  2 ;;
    -l)   			  loss="$2"      	; shift  2 ;;
    -r)   			  rate="$2"      	; shift  2 ;;
    -w)   			  wait="$2"      	; shift  2 ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen." ;;
  esac
done

sleep ${wait}

echo "TC apply ${eth}"

tc qdisc add dev ${eth} root handle 1: tbf rate ${rate}
tc qdisc add dev ${eth} parent 1:1 handle 10: netem delay ${delay} loss ${loss}
