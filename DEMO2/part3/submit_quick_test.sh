#!/bin/bash
# Quick sanity/demo job: 1 epoch + inference, no mixed precision.
# This satisfies DAWNBench requirement 2 ("run inference and a single epoch of
# training on Rangpur during the demonstration").
#
# Submit with:  sbatch submit_quick_test.sh
# Check status: squeue -u $USER
# Watch output: tail -f quick_test_<jobid>.out

#SBATCH --job-name=resnet18_quick
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=quick_test_%j.out
#SBATCH --error=quick_test_%j.err

echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# $HOME is shared across every node in the cluster (unlike /tmp, which is node-local) --
# store the dataset there so it only needs to be downloaded once, regardless of which
# node this job (or a later one) lands on.
python3 train_resnet_cifar.py --epochs 1 --batch-size 512 --data-root "$HOME/cifar_data"
