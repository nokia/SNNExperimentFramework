#!/bin/bash

eth=eth0
wait=0

PARSED_ARGUMENTS=$(getopt -a -n tc_down -o i:w: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
		echo "Invalid arguments"
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -i)   			  eth="$2"      	; shift  2 ;;
    -w)   			  wait="$2"      	; shift  2 ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen." ;;
  esac
done

sleep ${wait}

echo "TC down ${eth}"

tc qdisc del dev ${eth} root
