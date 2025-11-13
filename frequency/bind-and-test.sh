#!/bin/bash
# Usage: bind-and-test <core> <cmd_and_args>

if [[ $# -lt 2 ]] ; then
	echo "Usage: bind-and-test <core> <cmd_and_args>"
	exit
fi

core=$1
cmd_and_args=("${@:2}")
THRESHOLD=3900

freq() {
	core=$1
	freq=`cat /proc/cpuinfo | grep -A 10 "^processor\s*:\s*${core}$" | grep MHz | awk '{print $4}'`
	echo $freq
}

f=`freq ${core}`
while [[ `echo "$f < ${THRESHOLD}" | bc` -eq 1 ]] ; do
	echo "warming up: freq @ $f MHz"
	taskset -c ${core} stress --cpu 1 --timeout 3
	f=`freq ${core}`
done
echo "warmed up: freq @ $f MHz"

taskset -c ${core} ${cmd_and_args[@]}

