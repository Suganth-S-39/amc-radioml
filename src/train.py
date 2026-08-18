"""Training loop for AMC models.
Usage:
    python src/train.py --model multiscale --seed 42
    python src/train.py --model baseline --seed 1 --epochs 60
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import make_loaders, normalize_snr
from models import MODELS

def train(model_name, seed, epochs=60, batch_size=64, lr=1e-3,lambda_snr=0.1, data_dir='data/processed',
          out_dir='results/seeds', device=None):
    
    """Train one model for one seed. Returns best validation accuracy."""

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, _, (train_ds, val_ds, _) = make_loaders(data_dir, batch_size)

    model = MODELS[model_name]().to(device)
    criterion = nn.CrossEntropyLoss()
    criterion_snr = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    ckpt = os.path.join(out_dir, f'{model_name}_seed{seed}.pt')
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X, y, snr in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits, snr_pred = model(X)
            loss = criterion(logits, y)
            if snr_pred is not None:
                target = normalize_snr(snr.to(device))
                loss = loss + lambda_snr * criterion_snr(snr_pred.squeeze(), target)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)

        train_loss = running / len(train_ds)

        model.eval()
        val_running, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y, snr in val_loader:
                X, y = X.to(device), y.to(device)
                logits, _ = model(X)
                val_running += criterion(logits, y).item() * X.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        val_loss = val_running / len(val_ds)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)
            mark = ' <- saved'
        else:
            mark = ''

        lr_now = optimizer.param_groups[0]['lr']
        print(f'[{model_name} s{seed}] epoch {epoch+1}/{epochs} | '
              f'train {train_loss:.4f} | val {val_loss:.4f} | '
              f'acc {val_acc:.4f} | lr {lr_now:.6f}{mark}')

    elapsed = (time.time() - t0) / 60
    print(f'\n{model_name} seed {seed}: best val {best_val_acc:.4f} '
          f'({elapsed:.1f} min)')

    with open(os.path.join(out_dir, f'{model_name}_seed{seed}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    return best_val_acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(MODELS), default='multiscale')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--out-dir', default='results/seeds')
    args = p.parse_args()
    train(args.model, args.seed, args.epochs, args.batch_size,args.lr, data_dir=args.data_dir, out_dir=args.out_dir)

if __name__ == '__main__':
    main()