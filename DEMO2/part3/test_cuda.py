"""Minimal GPU check — from the COMP3710 "Getting Started on Rangpur" guide, Section 4.

Run this as a Slurm batch job (see cuda_check.sh) so it lands on a GPU node.
On the login node or a CPU node, torch.cuda.is_available() is expected to be False.
"""
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count :", torch.cuda.device_count())
    print("Device name  :", torch.cuda.get_device_name(0))
