#!/bin/bash

# Minimal: clamp one core between 3.9GHz and 4.0GHz
# Usage: fix-core-freq.sh <core_id> 

set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <core_id>" 1>&2
	exit 1
fi

core="$1"
CPU_DIR="/sys/devices/system/cpu/cpu${core}"
CPF_DIR="$CPU_DIR/cpufreq"

if [[ $EUID -ne 0 ]]; then
	echo "Re-running with sudo..."
	exec sudo "$0" "$core"
fi

if [[ ! -d "$CPF_DIR" ]]; then
	echo "cpufreq not available for cpu${core}" 1>&2
	exit 2
fi

MIN=3900000
MAX=4000000

echo "$MIN" > "$CPF_DIR/scaling_min_freq"
echo "$MAX" > "$CPF_DIR/scaling_max_freq"

echo "cpu${core}: min=${MIN} kHz max=${MAX} kHz"


