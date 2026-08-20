"""
config.py - Central configuration for adversarial image protection system.

All hyperparameters, paths, and defaults live here so every experiment
is fully reproducible by editing a single file.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224          # Resize input to (IMAGE_SIZE x IMAGE_SIZE)
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# ImageNet normalisation constants (used by all torchvision pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# White-box / surrogate models used during perturbation generation
WHITEBOX_MODELS = ["resnet50", "mobilenet_v3_large"]

# Black-box / held-out models NOT used during attack generation
BLACKBOX_MODELS = ["efficientnet_b0"]

# All models used for evaluation (union of both)
ALL_EVAL_MODELS = WHITEBOX_MODELS + BLACKBOX_MODELS

# Top-k for confidence reporting
TOP_K = 5

# ---------------------------------------------------------------------------
# Adversarial attack hyperparameters
# ---------------------------------------------------------------------------
# L∞ budget (8/255 ≈ 0.031 is a standard ImageNet adversarial budget)
EPSILON      = 8 / 255        # Maximum per-pixel perturbation (L∞ norm)
STEP_SIZE    = 2 / 255        # PGD step size (α)
NUM_STEPS    = 40             # PGD iterations
RANDOM_START = True           # Start from a random point inside the ε-ball

# L2 budget (alternative constraint, used when --norm l2 is specified)
EPSILON_L2   = 3.0

# ---------------------------------------------------------------------------
# Ensemble / transferable attack loss weights
# ---------------------------------------------------------------------------
ALPHA = 1.0     # Weight for adversarial loss  (L_adv)
BETA  = 0.5     # Weight for perceptual loss   (L_perceptual / SSIM proxy)
GAMMA = 0.1     # Weight for regularisation    (L_reg / L2 norm of delta)

# ---------------------------------------------------------------------------
# Robustness experiment settings
# ---------------------------------------------------------------------------
JPEG_QUALITIES     = [90, 75, 50, 30]   # JPEG quality levels to test
RESIZE_FACTORS     = [0.9, 0.75, 0.5]  # Fraction of original size
CROP_FRACTIONS     = [0.9, 0.8]        # Center-crop to this fraction
BRIGHTNESS_FACTORS = [0.8, 1.2]        # Multiply pixel values by this
GAUSSIAN_NOISE_STD = [0.02, 0.05]      # Std of additive Gaussian noise

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
VIS_DIR         = os.path.join(RESULTS_DIR, "visualizations")
SAMPLE_DIR      = os.path.join(BASE_DIR, "sample_images")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
PERTURBATION_AMPLIFY = 10   # Multiply perturbation by this factor for display

# ---------------------------------------------------------------------------
# Benchmark / experiment settings
# ---------------------------------------------------------------------------
BENCHMARK_METHODS = ["random_noise", "pgd_single", "pgd_ensemble"]
NUM_BENCHMARK_IMAGES = 5    # How many sample images to use in benchmark

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
