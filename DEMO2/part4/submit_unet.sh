#!/bin/bash
#SBATCH --job-name=oasis_unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:40:00
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source $HOME/miniconda3/bin/activate
conda activate torch

python train_unet_oasis.py
