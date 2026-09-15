#!/bin/bash
# Quick live-inference demo (no training) -- for the "run inference live during the
# demonstration" requirement. Needs unet_oasis.pt (produced by submit_unet.sh) to exist first.
#SBATCH --job-name=unet_infer
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --output=unet_infer_%j.out
#SBATCH --error=unet_infer_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source $HOME/miniconda3/bin/activate
conda activate torch

python infer_unet_oasis.py
