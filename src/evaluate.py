"""Evaluation for AMC models: test accuracy, TTA, and per-SNR breakdown.
Usage:
    python src/evaluate.py --model multiscale --seeds 42 1 7
"""

import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score
from dataset import make_loaders
from models import MODELS

TTA_ANGLES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)

CLASSES = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

def rotate_phase(x, theta):
    """Rotate a batch of I/Q signals by a fixed angle."""
    c, s = np.cos(theta), np.sin(theta)
    I, Q = x[:, 0], x[:, 1]
    return torch.stack([I * c - Q * s, I * s + Q * c], dim=1)

@torch.no_grad()
def predict(model, loader, device, tta_angles=(0.0,)):
    """Run inference, averaging softmax probabilities over TTA angles."""
    model.eval()
    preds, labels, snrs = [], [], []

    for X, y, snr in loader:
        X = X.to(device)
        probs = 0
        for theta in tta_angles:
            logits, _ = model(rotate_phase(X, theta))
            probs = probs + torch.softmax(logits, dim=1)
        preds.extend(probs.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        snrs.extend(snr.numpy())

    return np.array(preds), np.array(labels), np.array(snrs)

def per_snr_metrics(preds, labels, snrs):
    """Accuracy, macro-F1 and macro-precision for each SNR level."""
    rows = []
    for snr_val in sorted(np.unique(snrs)):
        m = snrs == snr_val
        yt, yp = labels[m], preds[m]
        rows.append({
            'snr': float(snr_val),
            'accuracy': float((yp == yt).mean()),
            'macro_f1': float(f1_score(yt, yp, average='macro', zero_division=0)),
            'macro_precision': float(precision_score(yt, yp, average='macro', zero_division=0)),
        })
    return rows

def evaluate(model_name, seeds, data_dir='data/processed',
             ckpt_dir='results/seeds', out_dir='results/metrics', device=None):
    """Evaluate one model across seeds, with and without TTA."""

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    _, _, test_loader, _ = make_loaders(data_dir)

    plain_accs, tta_accs = [], []
    best_run = None

    for seed in seeds:
        model = MODELS[model_name]().to(device)
        model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, f'{model_name}_seed{seed}.pt'),
                       map_location=device))

        preds, labels, snrs = predict(model, test_loader, device)
        preds_tta, _, _ = predict(model, test_loader, device, TTA_ANGLES)

        acc = (preds == labels).mean()
        acc_tta = (preds_tta == labels).mean()
        plain_accs.append(acc)
        tta_accs.append(acc_tta)

        if best_run is None or acc > best_run['acc']:
            best_run = {'seed': seed, 'acc': acc,
                        'preds': preds, 'preds_tta': preds_tta,
                        'labels': labels, 'snrs': snrs}
        print(f'seed {seed}: test {acc*100:.2f}%  |  +TTA {acc_tta*100:.2f}%')

    print(f'\n{model_name} test      : {np.mean(plain_accs)*100:.2f} '
          f'± {np.std(plain_accs)*100:.2f}%')
    print(f'{model_name} test +TTA   : {np.mean(tta_accs)*100:.2f} '
          f'± {np.std(tta_accs)*100:.2f}%')

    rows = per_snr_metrics(best_run['preds_tta'], best_run['labels'], best_run['snrs'])
    rows_plain = per_snr_metrics(best_run['preds'], best_run['labels'], best_run['snrs'])

    print(f"\nPer-SNR breakdown (seed {best_run['seed']}):")
    print(f"  {'SNR':>5} | {'Acc':>6} | {'+TTA':>6} | {'F1':>6} | {'Prec':>6}")
    for r, rp in zip(rows, rows_plain):
        print(f"  {r['snr']:>5.0f} | {rp['accuracy']*100:6.2f} | "
              f"{r['accuracy']*100:6.2f} | {r['macro_f1']*100:6.2f} | "
              f"{r['macro_precision']*100:6.2f}")

    results = {
        'model': model_name,
        'seeds': list(seeds),
        'test_acc_mean': float(np.mean(plain_accs)),
        'test_acc_std': float(np.std(plain_accs)),
        'test_acc_tta_mean': float(np.mean(tta_accs)),
        'test_acc_tta_std': float(np.std(tta_accs)),
        'per_seed_acc': [float(a) for a in plain_accs],
        'per_seed_acc_tta': [float(a) for a in tta_accs],
        'best_seed': best_run['seed'],
        'per_snr': rows,
        'per_snr_no_tta': rows_plain,
    }

    path = os.path.join(out_dir, f'{model_name}_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {path}')
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(MODELS), default='multiscale')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 7])
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--ckpt-dir', default='results/seeds')
    p.add_argument('--out-dir', default='results/metrics')
    args = p.parse_args()
    evaluate(args.model, args.seeds, args.data_dir, args.ckpt_dir, args.out_dir)

if __name__ == '__main__':
    main()