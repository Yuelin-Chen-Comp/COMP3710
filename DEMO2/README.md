# DEMO2 — Pattern Recognition

Lab Demonstration 2 for COMP3710. See [`COMP3710_Lab_2_2026.pdf`](COMP3710_Lab_2_2026.pdf) for the
official task sheet.

Each `part*/` folder is self-contained: a fully executed Jupyter notebook (outputs, plots and all)
plus a `steps/` folder with any standalone scripts or extracted images used while developing it.

## Part 1 — Discrete Fourier Transform (1 mark) ✅

[`part1/Part1_DFT.ipynb`](part1/Part1_DFT.ipynb)

- Reconstructs a square wave from its Fourier series and studies the Gibbs phenomenon.
- Naive DFT vs. NumPy's FFT, with a discussion of why the algorithmic complexity difference
  ($O(N^2)$ vs. $O(N\log N)$) dominates everything else.
- `square_wave`, `square_wave_fourier`, and `naive_dft` re-implemented in PyTorch, including a
  vectorised-matmul GPU version of the DFT.
- Timing comparisons across problem sizes (NumPy naive/FFT vs. PyTorch CPU/GPU), run both on a
  laptop (Apple Silicon `mps`) and, as an appendix, on a real NVIDIA GPU (Google Colab, `cuda`).

## Part 2 — Eigenfaces (1 mark) ✅

[`part2/Part2_Eigenfaces.ipynb`](part2/Part2_Eigenfaces.ipynb)

- PCA (via SVD) on the LFW face dataset; eigenfaces visualised as a gallery.
- Compactness plot (cumulative explained variance vs. number of components).
- Random Forest classifier on the PCA-reduced features; per-class precision/recall/F1, with a
  discussion of the class-imbalance problem in LFW.
- A small self-contained 2D SVD/PCA demo (`part2/steps/svd_2d_demo.py`) used to build intuition
  before applying the same idea to 1850-dimensional face images.

## Part 3 — CNNs (5 marks)

### 3.1 CNN classifier (1 mark) ✅

[`part3/Part3_CNN.ipynb`](part3/Part3_CNN.ipynb)

- A small CNN (two 3x3 conv layers, 32 filters each, + dense layers) trained end-to-end on the
  same LFW faces, in PyTorch.
- 88.2% test accuracy vs. 61% for the Part 2 PCA + Random Forest pipeline, evaluated on the
  identical held-out test set for a fair comparison.
- Training/test accuracy curves showing (and discussing) overfitting.

### 3.2 DAWNBench challenge — ResNet-18 on CIFAR-10 (4 marks) ✅ run on Rangpur A100

[`part3/Part3_2_DAWNBench.ipynb`](part3/Part3_2_DAWNBench.ipynb) — architecture, training
function, and a local correctness smoke test (small subset, few epochs).

[`part3/train_resnet_cifar.py`](part3/train_resnet_cifar.py) — the CLI version that ran on
Rangpur (full dataset, `cuda`, `--amp` mixed precision, OneCycle LR, label smoothing).

[`part3/cuda_check.sh`](part3/cuda_check.sh), [`part3/submit_quick_test.sh`](part3/submit_quick_test.sh),
[`part3/submit_full_dawnbench.sh`](part3/submit_full_dawnbench.sh) — SLURM job scripts for the
`comp3710` partition (`--account=comp3710` required; conda env activated in-script).

[`part3/dawnbench_587046.out`](part3/dawnbench_587046.out) and the other `*.out` files — the actual
Rangpur run logs.

ResNet-18 is implemented from scratch (no pretrained/pre-built model), CIFAR-style stem.
**Result on an A100 (mixed precision):** best run **93.49% test accuracy in 371 s**, 24 epochs.
Requirement 1 (>90%, fast) met comfortably; requirement 2 (live inference + 1 epoch on Rangpur) is
demonstrated live; requirement 3 (94% in ~360 s) is close on time, ~0.5% short on accuracy —
closing that last bit reliably needs a longer schedule or the full DAWNBench "bag of tricks"
beyond a plain ResNet-18.

## Part 4 — Recognition (8 marks)

### 4.1 Advanced Git Course (1 mark) ⏳ not code — external edX course

Not a coding task — complete the "Version Control for Teams using Git" short course on edX (see
the course's Blackboard/Ed Discussion post for the current enrolment link).

### 4.2–4.4 Recognition tasks (up to 7 marks, Hard tier — all three tasks attempted)

All three tasks use the **Preprocessed OASIS** brain MRI dataset, which only exists at
`/home/groups/comp3710/` on Rangpur — not available locally. Each task is therefore built and
verified locally on a stand-in dataset (documented in each notebook) first, with a "porting to
Rangpur" section spelling out exactly what changes for the real data. **All three are code-complete
and locally verified; all three still need a real Rangpur/OASIS run** for full marks (the actual
target metrics — DSC>0.9 on real brain tissue, OASIS-realistic GAN samples — can only be assessed
on the real dataset, and the live demo requirement needs Rangpur regardless).

- **Task 1 — VAE (Easy, 3 marks)** ✅ locally verified on MNIST
  [`part4/Part4_Task1_VAE.ipynb`](part4/Part4_Task1_VAE.ipynb) — reconstructions, a 2D
  latent-space scatter plot, and the required manifold visualisation (decoding a grid of latent
  points) all included and discussed.
- **Task 2 — UNet segmentation (+2 marks, Medium)** ✅ locally verified on synthetic shapes
  [`part4/Part4_Task2_UNet.ipynb`](part4/Part4_Task2_UNet.ipynb) — categorical (one-hot) output,
  per-class Dice score (>0.99 on the synthetic task; real OASIS will be a harder bar), and
  input/ground-truth/prediction visualisations.
- **Task 3 — GAN (+2 marks, Hard)** ✅ locally verified on MNIST (per the lab sheet's own
  suggestion to start there before OASIS)
  [`part4/Part4_Task3_GAN.ipynb`](part4/Part4_Task3_GAN.ipynb) — generator/discriminator training
  curves, and generated-sample snapshots across training showing no mode collapse.

## Environment notes

- Local development happened on a MacBook with no NVIDIA GPU — PyTorch's `mps` backend (Apple
  Silicon) stands in for `cuda` wherever a GPU comparison or GPU training run was needed locally.
  Anything that must run on real NVIDIA hardware is clearly marked "needs Rangpur" above.
- Rangpur access: SSH to `rangpur.compute.eait.uq.edu.au` with your UQ credentials; GPU jobs go
  through SLURM on the `comp3710` partition (`sinfo -p comp3710` to check node availability,
  `squeue -u $USER` to check your own jobs).
