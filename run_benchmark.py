#!/usr/bin/env python3
"""Quick benchmark runner: all methods x 3 images, 20 PGD steps."""
import torch, sys, os, json
sys.path.insert(0, '.')
import config
config.NUM_STEPS = 20   # fast but meaningful

from utils.preprocessing import load_image, save_image
from utils.visualization  import save_comparison, save_benchmark_table
from models.model_zoo      import get_all_models, predict, predict_class_index
from attacks.pgd           import pgd_attack
from attacks.ensemble      import ensemble_attack
from attacks.random_noise  import random_noise_attack
from metrics.image_quality import compute_all_metrics

torch.manual_seed(42)
device = 'cpu'

images = ['sample_images/img1.png', 'sample_images/img2.png', 'sample_images/img3.png']
os.makedirs('results/benchmark/images', exist_ok=True)
os.makedirs('results/benchmark/visualizations', exist_ok=True)

print('Loading all models ...')
models     = get_all_models(config.ALL_EVAL_MODELS, device=device)
surrogates = {n: models[n] for n in config.WHITEBOX_MODELS}

method_agg = {m: {'succ': [], 'conf': [], 'psnr': [], 'ssim': []}
              for m in ['random_noise', 'pgd_single', 'pgd_ensemble']}

for img_path in images:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    print(f'\n=== {stem} ===')
    x = load_image(img_path)

    # Original predictions
    orig = {}
    for name, model in models.items():
        cls, conf = predict_class_index(model, x, device=device)
        label = predict(model, x, top_k=1, device=device)[0][0]
        orig[name] = (cls, conf)
        print(f'  orig {name}: {label[:28]}  conf={conf:.4f}')

    for method in ['random_noise', 'pgd_single', 'pgd_ensemble']:
        if method == 'random_noise':
            x_adv = random_noise_attack(x, epsilon=8/255, device=device)
        elif method == 'pgd_single':
            x_adv = pgd_attack(surrogates[config.WHITEBOX_MODELS[0]], x,
                               epsilon=8/255, step_size=2/255, num_steps=20, device=device)
        else:
            x_adv = ensemble_attack(surrogates, x,
                                    epsilon=8/255, step_size=2/255, num_steps=20, device=device)

        m = compute_all_metrics(x.cpu(), x_adv.cpu())
        n_changed  = 0
        conf_total = 0.0
        for name, model in models.items():
            orig_cls, orig_conf = orig[name]
            prot_cls, prot_conf = predict_class_index(model, x_adv, device=device)
            if prot_cls != orig_cls:
                n_changed += 1
            conf_total += orig_conf - prot_conf

        succ   = n_changed / len(models)
        avg_cd = conf_total / len(models)
        psnr_v = m['psnr']
        ssim_v = m['ssim']
        print(f'  {method:<16} succ={succ*100:.0f}%  dconf={avg_cd:+.4f}  PSNR={psnr_v:.2f}  SSIM={ssim_v:.4f}')

        method_agg[method]['succ'].append(succ)
        method_agg[method]['conf'].append(avg_cd)
        method_agg[method]['psnr'].append(psnr_v)
        method_agg[method]['ssim'].append(ssim_v)

        save_image(x_adv, f'results/benchmark/images/{stem}_{method}.png')
        save_comparison(x.cpu(), x_adv.cpu(),
                        f'results/benchmark/visualizations/{stem}_{method}.png',
                        title=f'{stem} – {method}', metrics=m)

# ---- Aggregate ----
def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

print()
print('='*75)
print('BENCHMARK RESULTS TABLE')
print('='*75)
hdr = f"{'Method':<18}  {'Attack Success':>15}  {'Conf Reduction':>15}  {'PSNR (dB)':>10}  {'SSIM':>8}"
print(hdr)
print('-'*75)

rows = [{'method': 'original', 'attack_success_rate': 0.0,
         'avg_confidence_reduction': 0.0, 'avg_psnr': float('inf'), 'avg_ssim': 1.0}]

for method, a in method_agg.items():
    r = {
        'method':                    method,
        'attack_success_rate':       avg(a['succ']),
        'avg_confidence_reduction':  avg(a['conf']),
        'avg_psnr':                  avg(a['psnr']),
        'avg_ssim':                  avg(a['ssim']),
    }
    rows.append(r)
    asr    = r['attack_success_rate'] * 100
    dconf  = r['avg_confidence_reduction']
    psnr_v = r['avg_psnr']
    ssim_v = r['avg_ssim']
    print(f"{method:<18}  {asr:>14.1f}%  {dconf:>15.4f}  {psnr_v:>10.2f}  {ssim_v:>8.4f}")

print('='*75)

save_benchmark_table(rows, 'results/benchmark/benchmark_table.png')
with open('results/benchmark/results.json', 'w') as f:
    json.dump(rows, f, indent=2, default=str)
print('\nSaved results/benchmark/benchmark_table.png')
print('Saved results/benchmark/results.json')
