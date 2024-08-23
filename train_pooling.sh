#!/bin/bash
#
#SBATCH -M hpd
#SBATCH -J pool
#SBATCH --time 2-00:00:00
#SBATCH --mem 128G


module load gcc/11
export OMP_NUM_THREADS=8

python3.9 try_pooling_no_cuda.py
