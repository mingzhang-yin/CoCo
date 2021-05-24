#!/bin/sh
#
#SBATCH -A sml
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
##SBATCH --mail-user=
##SBATCH --mail-type=END


echo "python gmm.py --method ${METHOD} --seed ${SEED} --spurious True"
 
python gmm.py --method ${METHOD} --seed ${SEED} --spurious True


