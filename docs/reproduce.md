# Reproduce technical validation

[![English](https://img.shields.io/badge/Language-English-blue.svg)](reproduce.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](reproduce_zh.md)

**Prerequisites:** A complete [dataset root](assembly.md) (R0 + GitHub tag `v1.0.0`; merge restricted MP4s before re-running speech validation). Assembly: [assembly.md](assembly.md).

Run from the **dataset root** (directory containing `VERSION`).

All workflows accept `--quick` (small-sample smoke test, default) or `--full` (N=53).

## Traits

```powershell
conda activate ee_validation_speech
python code/validation_traits.py --quick
```

## Self-report

```powershell
python code/validation_selfreport.py --quick
```

## Speech (full re-run requires restricted AV merged)

To **re-run** the workflow, merge restricted MP4s first, then:

```powershell
conda env create -f environments/ee_validation_speech.yml
python code/validation_speech.py --full
```

## fNIRS quality

```powershell
python code/validation_fnirs_quality.py --quick
```

## fNIRS decoding

```powershell
conda env create -f environments/ee_validation_fnirs_decoding.yml
python code/validation_fnirs_decoding.py --full
```

Pre-shipped outputs under `results/validation_*/` are **manuscript-cited artifacts only** (tables, PDF figures, and selected stats CSVs per `.gitignore`). Full workflow re-runs write additional files locally. All validation counts use **released trials only** (see [data/README.md](../data/README.md#trial-inventory) and `data/subject_data_inventory.csv`).

See [install.md](install.md) for Homer3, FieldTrip, and OpenSMILE setup.
