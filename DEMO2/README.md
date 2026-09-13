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

### 4.2–4.4 Recognition tasks (up to 7 marks, Hard tier — all three tasks completed) ✅ run on real OASIS

All three tasks use the **Preprocessed OASIS** brain MRI dataset
(`/home/groups/comp3710/OASIS/keras_png_slices_*`, 256x256 PNG slices, 4-class segmentation masks
{0,85,170,255}). Each notebook first builds and verifies the approach on a stand-in dataset
(documented inline), then [`part4/oasis_dataset.py`](part4/oasis_dataset.py) +
`part4/train_{vae,unet,gan}_oasis.py` + `part4/submit_{vae,unet,gan}.sh` are the versions that
actually ran on Rangpur's A100s against the real dataset — logs and generated images are committed
alongside.

- **Task 1 — VAE (Easy, 3 marks)** ✅ **on real OASIS**: 30 epochs, 335s.
  [`part4/vae_recon.png`](part4/vae_recon.png) (reconstructions),
  [`part4/vae_latent_scatter.png`](part4/vae_latent_scatter.png) (2D latent space),
  [`part4/vae_manifold.png`](part4/vae_manifold.png) (the required manifold visualisation — a
  20x20 grid of decoded latent points, smoothly morphing from noise into distinct brain slices).
- **Task 2 — UNet segmentation (+2 marks, Medium)** ✅ **on real OASIS**: 25 epochs, 643s.
  **Test-set DSC: background 0.999, CSF 0.935, grey matter 0.943, white matter 0.967 — all four
  classes clear the >0.9 target.** Categorical (one-hot) output via per-pixel multi-class logits.
  [`part4/unet_predictions.png`](part4/unet_predictions.png),
  [`part4/unet_curves.png`](part4/unet_curves.png).
- **Task 3 — GAN (+2 marks, Hard)** ✅ **on real OASIS**: 60 epochs, 634s. Clear, non-collapsed
  brain-slice samples with varied ventricle shapes/sizes by the end of training; no mode collapse.
  [`part4/gan_progress.png`](part4/gan_progress.png) (generation quality over training),
  [`part4/gan_final_samples.png`](part4/gan_final_samples.png) (64 fresh samples),
  [`part4/gan_curves.png`](part4/gan_curves.png).

## Environment notes

- Local development happened on a MacBook with no NVIDIA GPU — PyTorch's `mps` backend (Apple
  Silicon) stands in for `cuda` wherever a GPU comparison or GPU training run was needed locally.
  Anything that must run on real NVIDIA hardware is clearly marked "needs Rangpur" above.
- Rangpur access: SSH to `rangpur.compute.eait.uq.edu.au` with your UQ credentials; GPU jobs go
  through SLURM on the `comp3710` partition (`sinfo -p comp3710` to check node availability,
  `squeue -u $USER` to check your own jobs).
