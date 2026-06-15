# External dependencies (fNIRS preprocessing and speech features)

[![English](https://img.shields.io/badge/Language-English-blue.svg)](install.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](install_zh.md)

## fNIRS (MATLAB)

- MATLAB R2023b (or compatible)
- [Homer3](https://github.com/BUNPC/Homer3) v1.80.2
- [FieldTrip](https://www.fieldtriptoolbox.org/) 20240110

The release ships preprocessing helpers under `code/utils/fnirs_preprocess/` but **not** the full Homer3 or FieldTrip trees (too large for the dataset package). Install them locally at the paths expected by `code/fnirs_preprocess.m`.

### 1. Install third-party toolboxes

From the **dataset root**:

```powershell
cd code\utils

# Homer3 v1.80.2
git clone --branch v1.80.2 --depth 1 https://github.com/BUNPC/Homer3.git Homer3-1.80.2

# FieldTrip 20240110 — download the release zip from fieldtriptoolbox.org,
# then extract/rename the folder to fieldtrip-20240110 here.
```

Expected layout:

```text
code/utils/
├── Homer3-1.80.2/
├── fieldtrip-20240110/
├── fnirs_preprocess/
└── homer3_patches/
```

### 2. Apply Homer3 patches

Two files must be patched before preprocessing (see also the header of `code/fnirs_preprocess.m`):

| Patch | Purpose |
|-------|---------|
| `hdf5write_safe.m` line 98: `H5T_NATIVE_ULONG` → `H5T_NATIVE_INT` | SNIRF save on macOS ([Homer3#181](https://github.com/BUNPC/Homer3/issues/181)) |
| `hmrR_PruneChannels.m` | Prune both wavelengths when either fails quality check |

Pre-patched copies live in `code/utils/homer3_patches/`. Apply them after cloning Homer3:

```powershell
.\scripts\apply_homer3_patches.ps1
```

Or copy manually:

```powershell
Copy-Item code\utils\homer3_patches\FuncRegistry\UserFunctions\hmrR_PruneChannels.m `
  code\utils\Homer3-1.80.2\FuncRegistry\UserFunctions\ -Force
Copy-Item code\utils\homer3_patches\DataTree\AcquiredData\DataFiles\Hdf5\hdf5write_safe.m `
  code\utils\Homer3-1.80.2\DataTree\AcquiredData\DataFiles\Hdf5\ -Force
```

### 3. Run preprocessing

In MATLAB, **cd to the dataset root** (directory containing `VERSION`), then:

```matlab
run('code/fnirs_preprocess.m')
```

The script adds paths, runs Homer3 `setpaths`, then FieldTrip `ft_defaults`. Raw `.nirs` files must be under `data/all_raw/{subject}/` with matching `*_rating.csv` trial logs.

Outputs go to `data/fnirs_signals/AEPO_001filt02/` (band-pass 0.01–0.2 Hz).

## Speech eGeMAPS (optional)

- [OpenSMILE](https://github.com/audeering/opensmile) 3.0 — Windows x64 build or `python-opensmile` via conda env `ee_validation_speech`

HuBERT features require GPU and packages listed in `environments/ee_validation_speech.yml`.

## Conda environments

| File | Purpose |
|------|---------|
| `environments/ee_experiments.yml` | PsychoPy experiment replication |
| `environments/ee_validation_speech.yml` | Speech validation |
| `environments/ee_validation_fnirs_decoding.yml` | fNIRS MIL decoding |

```powershell
conda env create -f environments/ee_validation_speech.yml
```
