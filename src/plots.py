"""Generate paper figures from saved evaluation results.
Usage:
    python src/plots.py --model multiscale
"""
import argparse
import json
import os
import matplotlib.pyplot as plt

def plot_metrics_vs_snr(results, out_path):
    snrs = [r['snr'] for r in results['per_snr']]
    acc = [r['accuracy'] * 100 for r in results['per_snr']]
    f1 = [r['macro_f1'] * 100 for r in results['per_snr']]
    prec = [r['macro_precision'] * 100 for r in results['per_snr']]

    plt.figure(figsize=(11, 6))
    plt.plot(snrs, acc,  'o-',  color='blue',  label='Accuracy',  linewidth=1.8)
    plt.plot(snrs, f1,   's--', color='green', label='F1 Score',  linewidth=1.8)
    plt.plot(snrs, prec, '^-.', color='red',   label='Precision', linewidth=1.8)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Score (%)')
    plt.xticks(snrs, rotation=45)
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, alpha=0.4)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')

def print_latex_table(results):
    """Print Table IV rows ready to paste into the paper."""
    print('\nLaTeX rows for the per-SNR table:\n')
    for r in results['per_snr']:
        print(f"${int(r['snr'])}$ & {r['accuracy']*100:.2f} & "
              f"{r['macro_f1']*100:.2f} & {r['macro_precision']*100:.2f} \\\\")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='multiscale')
    p.add_argument('--metrics-dir', default='results/metrics')
    p.add_argument('--fig-dir', default='results/figures')
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    with open(os.path.join(args.metrics_dir, f'{args.model}_results.json')) as f:
        results = json.load(f)

    plot_metrics_vs_snr(results,
                        os.path.join(args.fig_dir, 'metrics_vs_snr_corrected.png'))
    print_latex_table(results)

if __name__ == '__main__':
    main()