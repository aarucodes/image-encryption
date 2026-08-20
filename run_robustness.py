#!/usr/bin/env python3
"""Test robustness of PGD-protected image against common transformations."""
import torch, sys, os
sys.path.insert(0, '.')
import config

from utils.preprocessing import (
    load_image, apply_jpeg_compression, apply_resize_transform,
    apply_gaussian_noise, apply_brightness, apply_center_crop
)
from models.model_zoo import load_model, predict_class_index

torch.manual_seed(42)
device = 'cpu'

# Use already-generated PGD protected image
orig_path = 'sample_images/img1.png'
prot_path = 'results/benchmark/images/img1_pgd_single.png'

model = load_model('resnet50', device=device)

x_orig = load_image(orig_path)
x_prot = load_image(prot_path)

orig_cls, orig_conf = predict_class_index(model, x_orig, device=device)
prot_cls, prot_conf = predict_class_index(model, x_prot, device=device)

from models.model_zoo import predict
orig_label = predict(model, x_orig, top_k=1)[0][0]
prot_label = predict(model, x_prot, top_k=1)[0][0]

print(f'Original:  cls={orig_cls} ({orig_label[:30]}) conf={orig_conf:.4f}')
print(f'Protected: cls={prot_cls} ({prot_label[:30]}) conf={prot_conf:.4f}')
print(f'Protection active: {orig_cls != prot_cls}')
print()
print(f"{'Transform':<35} {'Prediction':<30} {'Same as orig?':>14} {'Prot survives?':>15}")
print('-'*100)

transforms = [
    ('jpeg_q90',        apply_jpeg_compression(x_prot, quality=90)),
    ('jpeg_q75',        apply_jpeg_compression(x_prot, quality=75)),
    ('jpeg_q50',        apply_jpeg_compression(x_prot, quality=50)),
    ('jpeg_q30',        apply_jpeg_compression(x_prot, quality=30)),
    ('resize_90pct',    apply_resize_transform(x_prot, factor=0.9)),
    ('resize_75pct',    apply_resize_transform(x_prot, factor=0.75)),
    ('resize_50pct',    apply_resize_transform(x_prot, factor=0.5)),
    ('brightness_80pct',apply_brightness(x_prot, factor=0.8)),
    ('brightness_120pct',apply_brightness(x_prot, factor=1.2)),
    ('gaussian_std002', apply_gaussian_noise(x_prot, std=0.02)),
    ('gaussian_std005', apply_gaussian_noise(x_prot, std=0.05)),
    ('crop_90pct',      apply_center_crop(x_prot, fraction=0.9)),
    ('crop_80pct',      apply_center_crop(x_prot, fraction=0.8)),
]

survived = 0
for name, x_t in transforms:
    t_cls, t_conf = predict_class_index(model, x_t.to(device), device=device)
    t_label = predict(model, x_t.to(device), top_k=1)[0][0]
    same_as_orig   = (t_cls == orig_cls)
    prot_survives  = (t_cls != orig_cls)
    if prot_survives:
        survived += 1
    marker = 'YES' if prot_survives else 'NO '
    print(f"{name:<35} {t_label[:29]:<30} {'yes' if same_as_orig else 'no':>14} {marker:>15}")

print()
print(f'Protection survival: {survived}/{len(transforms)} = {survived/len(transforms)*100:.0f}%')
