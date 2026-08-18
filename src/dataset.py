"""Dataset and preprocessing for RadioML 2016.10a."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RadioMLDataset(Dataset):
    """ RadioML I/Q samples with optional power normalization and phase-rotation augmentation.
        Power normalization is applied once at construction to all splits.
        Phase augmentation is applied per-sample at access time and should be enabled for training only.
    """

    def __init__(self, X, y, snr, normalize=True, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.snr = torch.from_numpy(snr).float()
        self.augment = augment

        if normalize:
            power = (self.X ** 2).sum(dim=1).mean(dim=1, keepdim=True)
            power = power.clamp(min=1e-8).sqrt().unsqueeze(-1)
            self.X = self.X / power

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.augment:
            theta = torch.rand(1) * 2 * np.pi
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            I, Q = x[0], x[1]
            x = torch.stack([I * cos_t - Q * sin_t,I * sin_t + Q * cos_t])
        return x, self.y[idx], self.snr[idx]

def load_splits(data_dir='data/processed'):
    """Load the pre-computed stratified splits from disk."""
    def _load(name):
        return np.load(f'{data_dir}/{name}.npy')

    return {
        'train': (_load('X_train'), _load('y_train'), _load('snr_train')),
        'val':   (_load('X_val'),   _load('y_val'),   _load('snr_val')),
        'test':  (_load('X_test'),  _load('y_test'),  _load('snr_test')),
    }

def make_loaders(data_dir='data/processed', batch_size=64):
    """ Build train/val/test DataLoaders with the standard configuration.
        Augmentation is enabled for training only.
    """
    splits = load_splits(data_dir)

    train_ds = RadioMLDataset(*splits['train'], normalize=True, augment=True)
    val_ds   = RadioMLDataset(*splits['val'],   normalize=True, augment=False)
    test_ds  = RadioMLDataset(*splits['test'],  normalize=True, augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
        (train_ds, val_ds, test_ds),
    )

# SNR normalization helpers for the FiLM model's auxiliary regression loss
SNR_MIN, SNR_MAX = -20.0, 18.0
SNR_MEAN = (SNR_MAX + SNR_MIN) / 2.0
SNR_SCALE = (SNR_MAX - SNR_MIN) / 2.0

def normalize_snr(snr):
    return (snr - SNR_MEAN) / SNR_SCALE

def denormalize_snr(snr_norm):
    return snr_norm * SNR_SCALE + SNR_MEAN