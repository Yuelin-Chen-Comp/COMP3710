#!/bin/bash
# GPU "hello world" — from the COMP3710 "Getting Started on Rangpur" guide, Section 4.
# Confirms the conda `torch` env works on a GPU node and that CUDA is visible.
#
# Submit with:  sbatch cuda_check.sh
# Check status: squeue --me
# Read output:  cat cuda_check_<jobid>.out

#SBATCH --job-name=cuda-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# A batch job starts from a clean environment, so activate conda here, not just in your shell.
source $HOME/miniconda3/bin/activate
conda activate torch

python test_cuda.py
