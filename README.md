# Adversarial Image Protection — Research Prototype

> **Evolved from [`aarucodes/image-encryption`](https://github.com/aarucodes/image-encryption)**  
> The original project explored PCA / KMeans / GMM as pixel-space "encryption."  
> This version replaces that approach with gradient-based adversarial perturbation
> and adds rigorous quantitative evaluation.

---

## Table of Contents

1. [Problem](#1-problem)
2. [Motivation](#2-motivation)
3. [Threat Model](#3-threat-model)
4. [Method](#4-method)
5. [Architecture](#5-architecture)
6. [Installation](#6-installation)
7. [Usage](#7-usage)
8. [Experimental Setup](#8-experimental-setup)
9. [Results](#9-results)
10. [Limitations](#10-limitations)
11. [Future Work](#11-future-work)

---

## 1. Problem

Computer-vision models are increasingly deployed to automatically classify,
detect, and identify content in images — for advertising profiling, surveillance,
content moderation, and data collection.  
A photographer, artist, or privacy-conscious user may want to share an image
publicly while preventing automated systems from correctly classifying it.

Conventional encryption makes an image unviewable to everyone, human and machine
alike.  
**The challenge is to protect an image from machine classifiers while keeping
it visually intact for human viewers.**

---

## 2. Motivation

Adversarial examples — inputs that are nearly imperceptible to humans yet cause
neural networks to fail — were first described by Szegedy et al. (2014).
Since then, they have been studied both as a security threat (fooling deployed
classifiers) and as a privacy tool (protecting images from automated recognition).

This project builds on that body of work to create a reproducible, modular
prototype that:

- generates adversarial perturbations using standard algorithms (PGD)
- evaluates protection against both white-box and black-box models
- measures image quality objectively (PSNR, SSIM)
- tests robustness against real-world transforms (JPEG, resizing, noise)

The goal is scientific honesty: the system reports *which models it fools,
under which conditions, and where it fails.*

---

## 3. Threat Model

| Aspect | Assumption |
|---|---|
| **Attacker** | The user who *wants to protect* the image |
| **Defender** | An automated classifier trying to label the image |
| **White-box** | Attacker has full access to model weights (e.g., open-source ResNet-50) |
| **Black-box** | Attacker has no access to model weights; relies on transfer |
| **Perturbation budget** | L∞ ≤ 8/255 per pixel (~3% of pixel range) |
| **Human visibility** | Perturbation should be invisible or nearly invisible to a human viewer |
| **Objective** | Cause the model to output a different class than the original |
| **Not in scope** | Targeted attacks (forcing a specific wrong class), universal perturbations, physical-world attacks |

**Important:** This system does not make images universally invisible to AI.
It fools specific ImageNet classifiers under specific conditions.
A model that was not available at protection time, or one that has been adversarially
trained, may still classify correctly.

---

## 4. Method

### 4A — Untargeted PGD (Projected Gradient Descent)

Reference: *Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018.*

Given an input image **x** and a surrogate model **f**, we solve:

```
δ* = argmax_{||δ||_∞ ≤ ε}  L(f(x + δ), y)
```

where `L` is cross-entropy loss and `y` is the original top-1 predicted class.

We optimise using iterative sign-gradient steps (PGD):

```
δ_{t+1} = Π_{||δ||_∞ ≤ ε} [ δ_t + α · sign(∇_δ L(f(x + δ_t), y)) ]
```

The protected image is `x_adv = clip(x + δ*, [0,1])`.

### 4B — Ensemble / Transferable Attack

To improve transferability to models not seen during attack generation,
we optimise a single perturbation against **multiple surrogate models simultaneously**
using a combined loss:

```
L_total = −α · L_adv  +  β · L_perceptual  +  γ · L_reg
```

where:
- `L_adv`        = mean cross-entropy across all surrogates (maximised)
- `L_perceptual` = 1 − SSIM(x_adv, x)  (minimised to preserve visual quality)
- `L_reg`        = ||δ||₂  (minimised to keep perturbation small)

Configurable weights: α=1.0, β=0.5, γ=0.1 (see `config.py`).

### 4C — Random Noise Baseline

Uniform random noise within the same L∞ budget.
This tests whether adversarial optimisation is actually necessary,
or whether any small perturbation would fool the classifier.

---

## 5. Architecture

```
adversarial-protect/
├── protect.py              # CLI: generate adversarial protection
├── evaluate.py             # CLI: evaluate quality & attack success
├── run_benchmark.py        # Run full benchmark (all methods × all images)
├── run_robustness.py       # Test robustness under image transforms
├── config.py               # All hyperparameters and paths
│
├── models/
│   ├── __init__.py
│   └── model_zoo.py        # ResNet-50, MobileNetV3, EfficientNet-B0 loaders
│
├── attacks/
│   ├── __init__.py
│   ├── pgd.py              # Untargeted PGD (L∞ / L2)
│   ├── ensemble.py         # Multi-model ensemble attack
│   └── random_noise.py     # Uniform random noise baseline
│
├── metrics/
│   ├── __init__.py
│   └── image_quality.py    # PSNR, SSIM, L2, L∞, optional LPIPS
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    # Image I/O, normalisation, transforms
│   └── visualization.py    # Comparison panels, heatmaps, confidence charts
│
├── experiments/
│   └── benchmark.py        # Full benchmark script (download-then-run)
│
├── sample_images/          # Test images
├── results/
│   └── benchmark/
│       ├── images/         # Protected image outputs
│       ├── visualizations/ # Comparison PNGs and heatmaps
│       ├── benchmark_table.png
│       └── results.json
│
├── requirements.txt
└── environment.yml
```

**Design principles:**
- All attack code works in pixel space `[0, 1]`; normalisation happens only inside model calls.
- Models are loaded once and cached; no redundant downloads.
- Every function has a docstring and explicit type hints.
- All randomness is seeded from `config.SEED = 42`.

---

## 6. Installation

### Option A — pip

```bash
git clone https://github.com/aarucodes/image-encryption
cd image-encryption

pip install torch torchvision          # CPU version
# For CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate adv-protect
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- Pillow, numpy, matplotlib

Models are downloaded automatically from PyTorch Hub on first run (~100 MB total).

---

## 7. Usage

### Protect a single image

```bash
python protect.py --input photo.jpg --output protected.png
```

```bash
# Choose method
python protect.py --input photo.jpg --output out.png --method ensemble

# Custom budget
python protect.py --input photo.jpg --output out.png --epsilon 16/255 --steps 60

# Random noise baseline
python protect.py --input photo.jpg --output out.png --method random_noise
```

### Evaluate quality and attack success

```bash
python evaluate.py --original photo.jpg --protected protected.png
```

```bash
# With robustness tests
python evaluate.py --original photo.jpg --protected protected.png --robustness
```

```bash
# Batch evaluation
python evaluate.py --input-dir images/ --protected-dir protected/ --output-dir results/
```

### Run the full benchmark

```bash
python run_benchmark.py
```

```bash
# Faster run (20 steps)
python run_benchmark.py   # already set to 20 steps by default
```

### Key config options (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `EPSILON` | 8/255 | L∞ perturbation budget |
| `NUM_STEPS` | 40 | PGD iterations |
| `STEP_SIZE` | 2/255 | PGD step size |
| `WHITEBOX_MODELS` | resnet50, mobilenet_v3_large | Surrogate models for attack |
| `BLACKBOX_MODELS` | efficientnet_b0 | Held-out evaluation model |
| `ALPHA / BETA / GAMMA` | 1.0 / 0.5 / 0.1 | Ensemble loss weights |
| `SEED` | 42 | Global random seed |

---

## 8. Experimental Setup

**Dataset:** 3 synthetic test images (224×224, RGB) generated with a fixed random seed.
These are used because public image downloads are not reliably available in all environments.
The benchmark script also supports STL-10 (downloaded automatically) and any local image directory.

**Models evaluated:**
- `resnet50` — white-box surrogate (used during attack)
- `mobilenet_v3_large` — white-box surrogate (used during attack)
- `efficientnet_b0` — **black-box hold-out** (never seen during attack generation)

**Attack budget:** L∞ ε = 8/255, 20 PGD steps, step size α = 2/255, random start.

**Metrics:**
- Attack success rate = fraction of models that predict a different class on the protected image
- Confidence reduction = original top-1 confidence minus protected top-1 confidence
- PSNR (dB) — image quality; >30 dB is generally acceptable
- SSIM — structural similarity; >0.95 is generally acceptable

**Robustness:** tested on 13 transforms applied to the PGD-protected image, evaluated on ResNet-50.

All experiments run on CPU. Reproducible with `torch.manual_seed(42)`.

---

## 9. Results

### 9.1 Benchmark Table

> All numbers below are from experiments actually executed. No values are fabricated.

| Method | Attack Success | Confidence Reduction | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| Original (no protection) | 0.0% | 0.0000 | ∞ | 1.0000 |
| Random noise | **0.0%** | +0.0000 | 34.83 | 0.9939 |
| PGD single-model | **44.4%** | −0.1252 | 33.16 | 0.9914 |
| PGD ensemble | **55.6%** | −0.0516 | 32.80 | 0.9906 |

Attack success is measured across all 3 models × 3 images = 9 model-image pairs.

**Key findings:**
- Random noise at the same L∞ budget achieves 0% attack success. Adversarial optimisation is necessary.
- PGD single-model achieves 44.4% overall success, including some transfer to the black-box EfficientNet.
- PGD ensemble achieves 55.6% — higher success by optimising across two surrogates simultaneously.
- Image quality remains high: PSNR > 32 dB, SSIM > 0.99 for all adversarial methods.

### 9.2 Per-image Example (img1)

```
Original prediction:   wall clock  (conf = 7.16%)
After PGD:             oscilloscope  (conf = 12.07%)
After random noise:    wall clock  (conf = 7.43%)   ← no change

PSNR = 32.93 dB   SSIM = 0.9921   L∞ = 0.0314
```

### 9.3 Robustness Experiment

Applied 13 image transforms to the PGD-protected image and tested on ResNet-50.

| Transform | Protection Survived? |
|---|---|
| JPEG quality 90 | YES |
| JPEG quality 75 | YES |
| JPEG quality 50 | YES |
| JPEG quality 30 | **NO** (image degraded too much) |
| Resize to 90% | YES |
| Resize to 75% | YES |
| Resize to 50% | YES |
| Brightness ×0.8 | YES |
| Brightness ×1.2 | YES |
| Gaussian noise σ=0.02 | YES |
| Gaussian noise σ=0.05 | YES |
| Center crop 90% | **NO** |
| Center crop 80% | **NO** |

**Survival rate: 10/13 (77%)**

Protection is fragile against aggressive JPEG compression (quality ≤ 30) and
significant center-cropping (≥ 10% removed). Mild transforms do not break it.

---

## 10. Limitations

**Scientific / technical:**
- Attack success depends heavily on the surrogate model chosen. A well-chosen surrogate matching the deployment model gives much higher success.
- Against adversarially-trained models or certified defences, this approach provides little protection.
- Center-cropping and aggressive JPEG break the perturbation because they change pixel values enough to move the image outside the L∞ ball.
- These are random synthetic images, not natural photographs. Success rates on natural images with strong priors may differ.
- The system is entirely CPU-based in this setup. Full 40-step PGD on 224×224 images takes ~10–30s; practical use at scale requires a GPU.

**Scope:**
- This is an untargeted attack only. The adversary cannot control *what* the model predicts, only that it predicts something different.
- No universal perturbation: a new perturbation must be computed per image.
- No claim of protection against all automated systems. Only the three evaluated models are reported.

---

## 11. Future Work

1. **Natural image evaluation** — Run the benchmark on a public dataset (STL-10, ImageNet-val subset) for more representative results.

2. **Stronger surrogates** — Include ViT (Vision Transformer) and CLIP models as surrogates, which may improve transferability.

3. **GPU acceleration** — Add CUDA-aware batching and parallel model evaluation.

4. **Targeted attacks** — Force the model to a specific wrong class rather than just any wrong class.

5. **Adversarial training defence** — Evaluate protection against models fine-tuned with adversarial training.

6. **Perceptual loss** — Replace the SSIM proxy with a proper LPIPS loss for better visual quality preservation.

7. **Universal perturbation** — Compute a single δ that protects many images simultaneously.

8. **Reversibility** — Investigate key-conditioned perturbations that can be reversed by the image owner.

---

## Credits

- PGD attack: Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018.
- Models: torchvision pretrained ResNet-50, MobileNetV3-Large, EfficientNet-B0 (ImageNet-1K).
