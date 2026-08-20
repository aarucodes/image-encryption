#!/usr/bin/env python3
"""
evaluate.py — Evaluate protection quality between original and protected images.

Also includes robustness experiments testing whether protection survives
common image transformations (JPEG compression, resizing, etc.).

Usage:
    # Single-image evaluation
    python evaluate.py --original image.jpg --protected protected.png

    # Batch evaluation
    python evaluate.py --input-dir images/ --protected-dir protected/ --output-dir results/

    # With robustness tests
    python evaluate.py --original image.jpg --protected protected.png --robustness

    # Batch with robustness
    python evaluate.py --input-dir images/ --protected-dir protected/ --output-dir results/ --robustness
"""

import argparse
import os
import sys
import json
import logging
import csv
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.preprocessing import (
    load_image, normalize,
    apply_jpeg_compression, apply_resize_transform,
    apply_gaussian_noise, apply_brightness, apply_center_crop,
)
from utils.visualization import save_comparison, save_confidence_chart, save_benchmark_table
from models.model_zoo     import load_model, predict, predict_class_index, get_all_models
from metrics.image_quality import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    original_path: str,
    protected_path: str,
    model_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    run_robustness: bool = False,
    vis_dir: Optional[str] = None,
    save_vis: bool = True,
) -> Dict:
    """Evaluate one original–protected image pair.

    Returns a comprehensive result dictionary.
    """
    if model_names is None: model_names = config.ALL_EVAL_MODELS
    if device      is None: device      = config.DEVICE
    if vis_dir     is None: vis_dir     = config.VIS_DIR

    logger.info(f"Evaluating: {os.path.basename(original_path)} ↔ {os.path.basename(protected_path)}")

    x_orig = load_image(original_path)
    x_prot = load_image(protected_path)

    # Image quality metrics
    metrics = compute_all_metrics(x_orig.cpu(), x_prot.cpu())
    logger.info(
        f"  PSNR={metrics['psnr']:.2f}dB  SSIM={metrics['ssim']:.4f}  "
        f"L∞={metrics['linf']:.4f}"
    )

    # Model predictions
    models = get_all_models(model_names, device=device)
    per_model = {}
    for name, model in models.items():
        orig_preds = predict(model, x_orig, top_k=config.TOP_K, device=device)
        prot_preds = predict(model, x_prot, top_k=config.TOP_K, device=device)

        orig_cls, orig_conf = orig_preds[0]
        prot_cls, prot_conf = prot_preds[0]

        per_model[name] = {
            "original_class":       orig_cls,
            "original_confidence":  orig_conf,
            "protected_class":      prot_cls,
            "protected_confidence": prot_conf,
            "prediction_changed":   orig_cls != prot_cls,
            "confidence_reduction": orig_conf - prot_conf,
            "original_preds":       orig_preds,
            "protected_preds":      prot_preds,
        }
        status = "✓ changed" if orig_cls != prot_cls else "✗ same"
        logger.info(
            f"  {name:<25} {status}  "
            f"orig='{orig_cls[:20]}' ({orig_conf*100:.1f}%)  →  "
            f"prot='{prot_cls[:20]}' ({prot_conf*100:.1f}%)"
        )

    attack_success = sum(1 for v in per_model.values() if v["prediction_changed"]) / len(per_model)

    # Visualisations
    if save_vis:
        stem = os.path.splitext(os.path.basename(original_path))[0]
        save_comparison(x_orig, x_prot, os.path.join(vis_dir, f"{stem}_eval_comparison.png"), metrics=metrics)
        chart_data = {n: {"original": v["original_preds"], "protected": v["protected_preds"]}
                      for n, v in per_model.items()}
        save_confidence_chart(chart_data, os.path.join(vis_dir, f"{stem}_eval_confidence.png"))

    # Robustness experiments
    robustness = {}
    if run_robustness:
        logger.info("  Running robustness experiments …")
        robustness = run_robustness_tests(x_prot, models, per_model, device=device)

    result = {
        "original_path":      original_path,
        "protected_path":     protected_path,
        "metrics":            metrics,
        "per_model":          per_model,
        "attack_success_rate": attack_success,
        "robustness":         robustness,
    }
    return result


# ---------------------------------------------------------------------------
# Robustness tests
# ---------------------------------------------------------------------------

def run_robustness_tests(
    x_prot: torch.Tensor,
    models: Dict,
    baseline_per_model: Dict,
    device: str = "cpu",
) -> Dict:
    """Test whether adversarial protection survives common transformations.

    For each transform, we measure how many models STILL misclassify
    (i.e., protection survived the transform).
    """
    transformations = {}

    # JPEG compression
    for quality in config.JPEG_QUALITIES:
        name = f"jpeg_q{quality}"
        x_t = apply_jpeg_compression(x_prot.cpu(), quality=quality).to(device)
        transformations[name] = x_t

    # Resize
    for factor in config.RESIZE_FACTORS:
        name = f"resize_{int(factor*100)}pct"
        x_t = apply_resize_transform(x_prot.cpu(), factor=factor).to(device)
        transformations[name] = x_t

    # Brightness
    for bf in config.BRIGHTNESS_FACTORS:
        name = f"brightness_{int(bf*100)}pct"
        x_t = apply_brightness(x_prot.cpu(), factor=bf).to(device)
        transformations[name] = x_t

    # Gaussian noise
    for std in config.GAUSSIAN_NOISE_STD:
        name = f"gaussian_noise_std{std}"
        x_t = apply_gaussian_noise(x_prot.cpu(), std=std, seed=config.SEED).to(device)
        transformations[name] = x_t

    # Center crop
    for frac in config.CROP_FRACTIONS:
        name = f"center_crop_{int(frac*100)}pct"
        x_t = apply_center_crop(x_prot.cpu(), fraction=frac).to(device)
        transformations[name] = x_t

    results = {}
    for transform_name, x_t in transformations.items():
        survived = 0
        total    = 0
        for model_name, model in models.items():
            orig_cls    = baseline_per_model[model_name]["original_class"]
            prot_preds  = predict(model, x_t, top_k=1, device=device)
            prot_cls    = prot_preds[0][0]
            if prot_cls != orig_cls:
                survived += 1
            total += 1
        results[transform_name] = {
            "protection_survived_rate": survived / total if total > 0 else 0,
            "survived": survived,
            "total": total,
        }
        logger.info(
            f"    {transform_name:<30}  survival={survived}/{total} "
            f"({survived/total*100:.0f}%)"
        )

    return results


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    input_dir: str,
    protected_dir: str,
    output_dir: str,
    model_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    run_robustness: bool = False,
) -> List[Dict]:
    """Evaluate all matching image pairs in two directories.

    Pairs are matched by filename (same stem, any supported extension).
    """
    if model_names is None: model_names = config.ALL_EVAL_MODELS
    if device      is None: device      = config.DEVICE

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Find matching pairs
    orig_files = {
        os.path.splitext(f)[0]: os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTS
    }
    prot_files = {
        os.path.splitext(f)[0]: os.path.join(protected_dir, f)
        for f in os.listdir(protected_dir)
        if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTS
    }

    pairs = [(orig_files[s], prot_files[s]) for s in orig_files if s in prot_files]
    if not pairs:
        logger.warning("No matching image pairs found!")
        return []

    logger.info(f"Found {len(pairs)} image pairs.")
    all_results = []

    for orig_path, prot_path in pairs:
        try:
            result = evaluate_pair(
                original_path=orig_path,
                protected_path=prot_path,
                model_names=model_names,
                device=device,
                run_robustness=run_robustness,
                vis_dir=vis_dir,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {orig_path}: {e}")

    # Aggregate statistics
    if all_results:
        _save_aggregate_results(all_results, output_dir)

    return all_results


def _save_aggregate_results(results: List[Dict], output_dir: str):
    """Save aggregate CSV and JSON summaries."""
    rows = []
    for r in results:
        row = {
            "image": os.path.basename(r["original_path"]),
            "attack_success_rate": r["attack_success_rate"],
            "psnr":  r["metrics"]["psnr"],
            "ssim":  r["metrics"]["ssim"],
            "l2":    r["metrics"]["l2"],
            "linf":  r["metrics"]["linf"],
        }
        # Per-model fields
        for name, pm in r["per_model"].items():
            row[f"{name}_changed"]    = pm["prediction_changed"]
            row[f"{name}_conf_delta"] = pm["confidence_reduction"]
        rows.append(row)

    # CSV
    csv_path = os.path.join(output_dir, "results_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Summary CSV: {csv_path}")

    # JSON
    json_path = os.path.join(output_dir, "results_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Full JSON: {json_path}")

    # Print aggregate table
    n = len(results)
    avg_success = sum(r["attack_success_rate"]    for r in results) / n
    avg_psnr    = sum(r["metrics"]["psnr"]        for r in results) / n
    avg_ssim    = sum(r["metrics"]["ssim"]        for r in results) / n

    print("\n" + "="*60)
    print(f"BATCH EVALUATION SUMMARY  ({n} images)")
    print("="*60)
    print(f"  Avg attack success rate : {avg_success*100:.1f}%")
    print(f"  Avg PSNR                : {avg_psnr:.2f} dB")
    print(f"  Avg SSIM                : {avg_ssim:.4f}")
    print("="*60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial image protection quality",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Single-image mode
    parser.add_argument("--original",    help="Original image path (single-image mode)")
    parser.add_argument("--protected",   help="Protected image path (single-image mode)")

    # Batch mode
    parser.add_argument("--input-dir",   help="Directory of original images (batch mode)")
    parser.add_argument("--protected-dir", help="Directory of protected images (batch mode)")
    parser.add_argument("--output-dir",  default="results/", help="Output directory for batch mode")

    # Options
    parser.add_argument("--models",      nargs="+", default=None, help="Model names to evaluate")
    parser.add_argument("--device",      default=None,            help="cpu or cuda")
    parser.add_argument("--robustness",  action="store_true",     help="Run robustness experiments")
    parser.add_argument("--no-vis",      action="store_true",     help="Skip saving visualisations")
    parser.add_argument("--vis-dir",     default=None,            help="Visualisation output directory")
    parser.add_argument("--save-json",   default=None,            help="Save results JSON")

    args = parser.parse_args()

    if args.original and args.protected:
        # Single-image mode
        result = evaluate_pair(
            original_path  = args.original,
            protected_path = args.protected,
            model_names    = args.models,
            device         = args.device,
            run_robustness = args.robustness,
            vis_dir        = args.vis_dir,
            save_vis       = not args.no_vis,
        )

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print("\nImage Quality:")
        for k, v in result["metrics"].items():
            print(f"  {k.upper():8s} = {v:.4f}")
        print("\nPer-model:")
        print(f"  {'Model':<25} {'Changed':>8}  {'Orig Conf':>10}  {'Prot Conf':>10}  {'ΔConf':>8}")
        print(f"  {'-'*70}")
        for name, r in result["per_model"].items():
            print(
                f"  {name:<25} {'YES' if r['prediction_changed'] else 'no':>8}  "
                f"{r['original_confidence']:>10.4f}  "
                f"{r['protected_confidence']:>10.4f}  "
                f"{r['confidence_reduction']:>+8.4f}"
            )
        print(f"\nAttack success rate: {result['attack_success_rate']*100:.1f}%")

        if args.robustness and result["robustness"]:
            print("\nRobustness:")
            for t_name, t_res in result["robustness"].items():
                print(
                    f"  {t_name:<30}  "
                    f"survival={t_res['protection_survived_rate']*100:.0f}%"
                )

        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved: {args.save_json}")

    elif args.input_dir and args.protected_dir:
        # Batch mode
        evaluate_batch(
            input_dir      = args.input_dir,
            protected_dir  = args.protected_dir,
            output_dir     = args.output_dir,
            model_names    = args.models,
            device         = args.device,
            run_robustness = args.robustness,
        )
    else:
        parser.error("Provide either --original + --protected (single) or "
                     "--input-dir + --protected-dir (batch).")


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    main()
