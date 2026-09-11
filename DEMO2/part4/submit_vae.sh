#!/bin/bash
#SBATCH --job-name=oasis_vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:40:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source $HOME/miniconda3/bin/activate
conda activate torch

python train_vae_oasis.py
