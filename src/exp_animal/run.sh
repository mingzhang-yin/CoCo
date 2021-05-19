#!/bin/sh
#
#SBATCH -A sml
#SBATCH -c 6
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1


python main_coco.py --lr 0.003  --anneal 100 --factor 1e5
python main_irm.py --lr 0.003 --anneal 100 --lmbd 1e4
python main_irm.py --lr 0.003 --anneal 100 --lmbd 0
