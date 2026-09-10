#!/bin/bash
# Quick sanity/demo job: 1 epoch + inference, no mixed precision.
# This satisfies DAWNBench requirement 2 ("run inference and a single epoch of
# training on Rangpur during the demonstration").
#
# Submit with:  sbatch submit_quick_test.sh
# Check status: squeue --me
# Watch output: tail -f quick_test_<jobid>.out

#SBATCH --job-name=resnet18_quick
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:20:00
#SBATCH --output=quick_test_%j.out
#SBATCH --error=quick_test_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# A batch job starts from a clean environment, so activate the conda env here.
source $HOME/miniconda3/bin/activate
conda activate torch

# $HOME is shared across every node in the cluster (unlike /tmp, which is node-local) --
# store the dataset there so it only needs to be downloaded once, regardless of which
# node this job (or a later one) lands on.
python train_resnet_cifar.py --epochs 1 --batch-size 512 --data-root "$HOME/cifar_data"
