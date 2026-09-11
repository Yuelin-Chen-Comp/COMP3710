#!/bin/bash
# Full DAWNBench run: mixed precision, OneCycle LR, label smoothing + random erasing.
# A plain 20-epoch run reached 93.01% in 301s. This adds label smoothing (0.1) and Cutout
# (RandomErasing) and 4 more epochs to push past the 94% / ~360s target.
#
# Submit with:  sbatch submit_full_dawnbench.sh
# Check status: squeue --me
# Watch output: tail -f dawnbench_<jobid>.out

#SBATCH --job-name=resnet18_dawnbench
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:45:00
#SBATCH --output=dawnbench_%j.out
#SBATCH --error=dawnbench_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# A batch job starts from a clean environment, so activate the conda env here.
source $HOME/miniconda3/bin/activate
conda activate torch

# $HOME is shared across every node in the cluster (unlike /tmp, which is node-local) --
# store the dataset there so it only needs to be downloaded once, regardless of which
# node this job (or a later one) lands on.
python train_resnet_cifar.py --epochs 24 --batch-size 512 --max-lr 0.4 --amp \
    --label-smoothing 0.1 --data-root "$HOME/cifar_data"
