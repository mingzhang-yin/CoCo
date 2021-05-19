#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUT_SUFFIX=".out"
RUN_SCRIPT="run_base.sh"
Ms="CoCO IRM ERM"

for METHOD in ${Ms}; do
	export METHOD=${METHOD}
	for ((i=1;i<=5;i++)); do
		export SEED=$i
		export NAME=bash
		export OUTNAME=$i${OUT_SUFFIX}
		sbatch --job-name=${NAME} \
		--output=${OUTNAME} \
		--exclude=gonzo \
		${RUN_SCRIPT}
	done
done

