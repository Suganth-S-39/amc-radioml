# Automatic Modulation Classification at Low SNR

A compact multi-scale CNN-BiLSTM-attention network for classifying 11
modulation schemes from raw I/Q signals on RadioML 2016.10a.

## Results

| Model | Params | Test Acc. | Acc. @ 0 dB |
|---|---|---|---|
| CNN-BiLSTM-Attention (baseline) | 88,716 | 62.14 ± 0.49% | — |
| Multi-scale (proposed) | 95,532 | 62.83 ± 0.20% | 90.48% |
| Multi-scale + 4-angle TTA | 95,532 | **63.23 ± 0.18%** | **91.21%** |

All figures are means over three random seeds (42, 1, 7).

## Key findings

- **Preprocessing dominates architecture.** Power normalization and
  phase-rotation augmentation contribute +3.46 points; individual
  architectural changes contribute less than 0.7.
- **Measured variance floor is ~0.5 points.** Single-run improvements
  below this cannot be distinguished from seed noise. Even fixed-seed
  runs differ by ~0.2 points due to non-deterministic cuDNN kernels.
- **WBFM/AM-DSB confusion is structural.** WBFM reaches only 38.7% at
  0 dB and 43.3% at 18 dB — it does not recover with SNR. Excluding
  WBFM, accuracy at 0 dB is approximately 96.5%.

## Setup

```bash
conda create -n amc python=3.10 -y
conda activate amc
pip install -r requirements.txt
```

Download RadioML 2016.10a from [DeepSig](https://www.deepsig.ai/datasets)
and place `RML2016.10a_dict.pkl` in `data/`. Run
`notebooks/01_data_exploration.ipynb` to generate the stratified splits
in `data/processed/`.

## Usage

```bash
# train one model for one seed
python src/train.py --model multiscale --seed 42

# evaluate across seeds, with and without TTA
python src/evaluate.py --model multiscale --seeds 42 1 7

# regenerate figures from saved metrics
python src/plots.py --model multiscale
```

## Structure

```
src/
  dataset.py     RadioML dataset, normalization, phase augmentation
  models.py      baseline, multi-scale, and FiLM architectures
  train.py       training loop with LR scheduling and checkpointing
  evaluate.py    test evaluation, TTA, per-SNR breakdown
  plots.py       figure generation
notebooks/       exploratory work, in order
results/
  seeds/         model checkpoints
  metrics/       evaluation output (JSON)
  figures/       paper figures
```

## Dataset

RadioML 2016.10a: 220,000 samples, 11 modulation classes, 20 SNR levels from −20 to +18 dB.
Each sample is a 2×128 array of in-phase and quadrature components.
Split 70/15/15, stratified jointly on modulation and SNR.