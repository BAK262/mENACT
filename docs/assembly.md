# Assemble a complete dataset root

[![English](https://img.shields.io/badge/Language-English-blue.svg)](assembly.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](assembly_zh.md)

A **dataset root** is one folder containing `VERSION`, `data/`, and `code/` (and related engineering trees). This guide is the **only** place with full assembly steps.

**Prerequisites:** You want paper workflows from [reproduce.md](reproduce.md). For neuro-only or video-only use, see [use_cases/](use_cases/).

## Steps

### 1. Open data (Zenodo R0)

Download [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) and unzip to your dataset root (folder that will contain `VERSION` and `data/`).

### 2. Engineering tree (GitHub)

From the dataset root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/assemble_dataset.ps1 -Root .
```

Or manually clone tag `v1.0.0` and move `code/`, `experiments/`, `environments/`, `docs/`, `scripts/`, and `results/` into the root.

Repository: https://github.com/BAK262/mENACT

### 3. Optional — restricted MP4s and scripts

1. Read [data_use_agreement.md](data_use_agreement.md), then request access via [access_request.md](access_request.md).
2. Download zip file(s) using the download link(s) emailed to you.
3. Merge:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/restore_restricted.ps1 -Root . -Parts ".\downloads\menact_v1.0.0_restricted_*.zip"
```

### 4. Environment

```powershell
conda env create -f environments/ee_validation_speech.yml
```

Details: [install.md](install.md)

### 5. Verify

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_layout.ps1 -Root . -Profile full
python code/validation_traits.py --quick
```

## Product reference

[downloads.md](downloads.md)
