#!/usr/bin/env python3
"""
protect.py — Adversarial Image Protection CLI

Produces a visually similar but adversarially perturbed version of an input image
that causes selected computer-vision models to misclassify it.

Usage examples:
    # Single image, PGD against ResNet-50
    python protect.py --input image.jpg --output protected.png

    # Specify method and epsilon
    python protect.py --input image.jpg --output out.png --method pgd --epsilon 0.03

    # Ensemble attack against multiple surrogates
    python protect.py --input image.jpg --output out.png --method ensemble

    # Random noise baseline
    python protect.py --input image.jpg --output out.png --method random_noise

    # Custom epsilon and iterations
    python protect.py --input image.jpg --output out.png \\
        --epsilon 16/255 --steps 60 --step-size 2/255
"""

import argparse
import os
import sys
import logging
import json
import time
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.preprocessing import load_image, save_image
from utils.visualization  import save_comparison, save_perturbation_map, save_confidence_chart
from models.model_zoo      import load_model, predict, get_all_models
from attacks.pgd           import pgd_attack
from attacks.ensemble      import ensemble_attack
from attacks.random_noise  import random_noise_attack
from metrics.image_quality import compute_all_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_epsilon(value: str) -> float:
    """Accept either a float string ('0.031') or a fraction ('8/255')."""
    if "/" in value:
        num, den = value.split("/")
        return float(num) / float(den)
    return float(value)


def _summarise_predictions(preds):
    """Return a compact string of top-3 predictions."""
    return "  ".join(f"{cls[:18]}({conf*100:.1f}%)" for cls, conf in preds[:3])


# ---------------------------------------------------------------------------
# Core protection function
# ---------------------------------------------------------------------------

def protect_image(
    input_path: str,
    output_path: str,
    method: str = "pgd",
    epsilon: float = None,
    step_size: float = None,
    num_steps: int = None,
    norm: str = "linf",
    surrogate_models: list = None,
    device: str = None,
    save_vis: bool = True,
    vis_dir: str = None,
    verbose: bool = False,
) -> dict:
    """Protect a single image and return a result dictionary.

    Args:
        input_path:        Path to source image.
        output_path:       Where to save the protected image.
        method:            'pgd', 'ensemble', or 'random_noise'.
        epsilon:           L∞ perturbation budget.
        step_size:         PGD step size.
        num_steps:         PGD iterations.
        norm:              'linf' or 'l2'.
        surrogate_models:  Model names used during perturbation generation.
        device:            Torch device.
        save_vis:          Whether to save comparison visualisations.
        vis_dir:           Directory to save visualisations.
        verbose:           Print progress.

    Returns:
        Result dict with predictions, metrics, and file paths.
    """
    # Defaults
    if epsilon         is None: epsilon         = config.EPSILON
    if step_size       is None: step_size       = config.STEP_SIZE
    if num_steps       is None: num_steps       = config.NUM_STEPS
    if surrogate_models is None: surrogate_models = config.WHITEBOX_MODELS
    if device          is None: device          = config.DEVICE
    if vis_dir         is None: vis_dir         = config.VIS_DIR

    # ---- Load image ----
    logger.info(f"Loading image: {input_path}")
    x = load_image(input_path)
    logger.info(f"  Image size: {x.shape[2]}×{x.shape[3]}")

    # ---- Load surrogate models ----
    logger.info(f"Loading surrogate models: {surrogate_models}")
    models = {name: load_model(name, device=device) for name in surrogate_models}

    # ---- Evaluate original image ----
    logger.info("Evaluating original image …")
    all_model_names = config.ALL_EVAL_MODELS
    all_models      = get_all_models(all_model_names, device=device)

    original_preds = {}
    for name, model in all_models.items():
        preds = predict(model, x, top_k=config.TOP_K, device=device)
        original_preds[name] = preds
        logger.info(f"  {name} → {_summarise_predictions(preds)}")

    # ---- Generate adversarial perturbation ----
    logger.info(f"Generating perturbation: method={method}, ε={epsilon:.4f}, steps={num_steps}")
    t0 = time.time()

    if method == "pgd":
        # Use first surrogate only
        main_surrogate = models[surrogate_models[0]]
        x_adv = pgd_attack(
            model=main_surrogate,
            x=x,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=num_steps,
            norm=norm,
            device=device,
            verbose=verbose,
        )
    elif method == "ensemble":
        x_adv = ensemble_attack(
            models=models,
            x=x,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=num_steps,
            norm=norm,
            device=device,
            verbose=verbose,
        )
    elif method == "random_noise":
        x_adv = random_noise_attack(x=x, epsilon=epsilon, device=device)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pgd', 'ensemble', or 'random_noise'.")

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s")

    # ---- Evaluate protected image ----
    logger.info("Evaluating protected image …")
    protected_preds = {}
    for name, model in all_models.items():
        preds = predict(model, x_adv, top_k=config.TOP_K, device=device)
        protected_preds[name] = preds
        logger.info(f"  {name} → {_summarise_predictions(preds)}")

    # ---- Compute image quality metrics ----
    metrics = compute_all_metrics(x.cpu(), x_adv.cpu(), device="cpu")
    logger.info(
        f"Quality metrics: PSNR={metrics['psnr']:.2f}dB  "
        f"SSIM={metrics['ssim']:.4f}  L∞={metrics['linf']:.4f}"
    )

    # ---- Save protected image ----
    save_image(x_adv, output_path)
    logger.info(f"Protected image saved: {output_path}")

    # ---- Visualisations ----
    if save_vis:
        stem = os.path.splitext(os.path.basename(input_path))[0]

        # Comparison panel
        comp_path = os.path.join(vis_dir, f"{stem}_{method}_comparison.png")
        save_comparison(
            original=x, protected=x_adv,
            output_path=comp_path,
            title=f"Adversarial Protection — {method} (ε={epsilon:.3f})",
            metrics=metrics,
        )
        logger.info(f"Comparison saved: {comp_path}")

        # Perturbation heatmap
        heat_path = os.path.join(vis_dir, f"{stem}_{method}_heatmap.png")
        save_perturbation_map(x, x_adv, output_path=heat_path)
        logger.info(f"Heatmap saved: {heat_path}")

        # Confidence chart
        chart_data = {
            name: {
                "original":  original_preds[name],
                "protected": protected_preds[name],
            }
            for name in all_model_names
        }
        chart_path = os.path.join(vis_dir, f"{stem}_{method}_confidence.png")
        save_confidence_chart(chart_data, output_path=chart_path)
        logger.info(f"Confidence chart saved: {chart_path}")

    # ---- Build result summary ----
    results = {}
    for name in all_model_names:
        orig_top1_cls,  orig_top1_conf  = original_preds[name][0]
        prot_top1_cls,  prot_top1_conf  = protected_preds[name][0]
        pred_changed = orig_top1_cls != prot_top1_cls

        results[name] = {
            "original_class":      orig_top1_cls,
            "original_confidence": orig_top1_conf,
            "protected_class":     prot_top1_cls,
            "protected_confidence": prot_top1_conf,
            "prediction_changed":  pred_changed,
            "confidence_reduction": orig_top1_conf - prot_top1_conf,
            "role": "white-box" if name in surrogate_models else "black-box",
        }

    summary = {
        "input":      input_path,
        "output":     output_path,
        "method":     method,
        "epsilon":    epsilon,
        "num_steps":  num_steps,
        "device":     device,
        "elapsed_s":  elapsed,
        "metrics":    metrics,
        "per_model":  results,
        "attack_success_rate": sum(
            1 for v in results.values() if v["prediction_changed"]
        ) / len(results),
    }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Image Protection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",      required=True,          help="Input image path")
    parser.add_argument("--output",     required=True,          help="Output protected image path")
    parser.add_argument("--method",     default="pgd",
                        choices=["pgd", "ensemble", "random_noise"],
                        help="Protection method (default: pgd)")
    parser.add_argument("--epsilon",    default=None,           help="L∞ budget, e.g. 8/255 or 0.031")
    parser.add_argument("--step-size",  default=None,           help="PGD step size, e.g. 2/255")
    parser.add_argument("--steps",      default=None, type=int, help="Number of PGD iterations")
    parser.add_argument("--norm",       default="linf",
                        choices=["linf", "l2"],
                        help="Perturbation norm (default: linf)")
    parser.add_argument("--models",     default=None, nargs="+",
                        help="Surrogate model names (default: from config)")
    parser.add_argument("--device",     default=None,           help="cpu or cuda (default: auto)")
    parser.add_argument("--no-vis",     action="store_true",    help="Skip saving visualisations")
    parser.add_argument("--vis-dir",    default=None,           help="Visualisation output directory")
    parser.add_argument("--save-json",  default=None,           help="Save results JSON to this path")
    parser.add_argument("--verbose",    action="store_true",    help="Verbose attack progress")

    args = parser.parse_args()

    # Parse epsilon and step-size fractions
    epsilon   = _parse_epsilon(args.epsilon)   if args.epsilon   else None
    step_size = _parse_epsilon(args.step_size) if args.step_size else None

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    summary = protect_image(
        input_path       = args.input,
        output_path      = args.output,
        method           = args.method,
        epsilon          = epsilon,
        step_size        = step_size,
        num_steps        = args.steps,
        norm             = args.norm,
        surrogate_models = args.models,
        device           = args.device,
        save_vis         = not args.no_vis,
        vis_dir          = args.vis_dir,
        verbose          = args.verbose,
    )

    # ---- Print results ----
    print("\n" + "="*60)
    print("PROTECTION RESULTS")
    print("="*60)
    print(f"Method:   {summary['method']}")
    print(f"Epsilon:  {summary['epsilon']:.4f}  ({summary['epsilon']*255:.1f}/255)")
    print(f"Steps:    {summary['num_steps']}")
    print(f"Device:   {summary['device']}")
    print(f"Time:     {summary['elapsed_s']:.1f}s")
    print()
    print("Image Quality:")
    for k, v in summary["metrics"].items():
        print(f"  {k.upper():8s} = {v:.4f}")
    print()
    print("Per-model Results:")
    print(f"  {'Model':<25} {'Role':>10}  {'Original Class':<25} {'Protected Class':<25} {'Changed':>8}  {'ΔConf':>8}")
    print(f"  {'-'*110}")
    for model_name, r in summary["per_model"].items():
        print(
            f"  {model_name:<25} {r['role']:>10}  "
            f"{r['original_class'][:24]:<25} {r['protected_class'][:24]:<25} "
            f"{'YES' if r['prediction_changed'] else 'no':>8}  "
            f"{r['confidence_reduction']:>+8.4f}"
        )
    print()
    print(f"Attack success rate (all models): {summary['attack_success_rate']*100:.1f}%")
    print("="*60)

    # Save JSON if requested
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {args.save_json}")

    return summary


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    main()
