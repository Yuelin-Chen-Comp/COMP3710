#!/bin/bash
# Full DAWNBench run: mixed precision + more epochs, aiming for the >90%/94% targets.
# Tune --epochs and --max-lr based on what you see from the quick test job first --
# these starting values are a reasonable guess, not guaranteed to hit the target as-is.
#
# Submit with:  sbatch submit_full_dawnbench.sh
# Check status: squeue -u $USER
# Watch output: tail -f dawnbench_<jobid>.out

#SBATCH --job-name=resnet18_dawnbench
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=dawnbench_%j.out
#SBATCH --error=dawnbench_%j.err

echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 train_resnet_cifar.py --epochs 20 --batch-size 512 --max-lr 0.4 --amp
