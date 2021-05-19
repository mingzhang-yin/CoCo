#!/bin/sh
#
#SBATCH -A sml
#SBATCH --cpus-per-task=4
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:1


echo "python mnist.py --method ${METHOD} --seed ${SEED}"
 
python mnist.py --method ${METHOD} --seed ${SEED}


