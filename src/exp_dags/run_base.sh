#!/bin/sh
#
#SBATCH -A sml
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
##SBATCH --mail-user=mzyin@utexas.edu
##SBATCH --mail-type=END


echo "python dag.py --max_iter 30000 --seed ${SEED}"
 
python dag.py --max_iter 30000 --seed ${SEED} 


