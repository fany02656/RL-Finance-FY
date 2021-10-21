#!/bin/bash

#SBATCH --job-name=american_hedging_train_parallel
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,ARRAY_TASKS
#SBATCH --mail-user=ars991@nyu.edu
#SBATCH --nodes=1 --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512MB
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ars991/logs/american_hedging_cts_%A_%a.out
#SBATCH --error=/scratch/ars991/logs/american_hedging_cts_%A_%a.err
#SBATCH --array=0-511

module purge
module load python/intel/3.8.6

cd /scratch/ars991/RL-Finance
python train_parallel.py $SLURM_ARRAY_TASK_ID