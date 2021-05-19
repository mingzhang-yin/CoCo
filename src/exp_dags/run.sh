#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUT_SUFFIX=".out"
RUN_SCRIPT="run_base.sh"

for ((i=1;i<=10;i++)); do
	export SEED=$i
	export NAME=bash
	export OUTNAME=$i${OUT_SUFFIX}
	sbatch --job-name=${NAME} \
	--output=${OUTNAME} \
	${RUN_SCRIPT}
done

