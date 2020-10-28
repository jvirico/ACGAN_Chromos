#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -p gpu	  # use the GPU partition
#SBATCH -p gpu_lowpriority
#SBATCH --gres=gpu:v100:1	#select 1 V100 GPU
#SBATCH -c 24      # cores requested
#SBATCH -o outfile-%j  # send stdout to outfile
#SBATCH -e errfile-%j  # send stderr to errfile

#squeue
#scancel <job_ID>

echo "Starting job"
module load python/3.6
python ../acgan.py
echo "Finishing job"