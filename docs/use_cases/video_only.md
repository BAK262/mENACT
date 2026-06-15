# Participant video only

[![English](https://img.shields.io/badge/Language-English-blue.svg)](video_only.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](video_only_zh.md)

Use this path for **MP4-only** analysis (e.g. behavior or vision modeling). You do **not** need GitHub or Zenodo R0 unless you want trial alignment with fNIRS/behavior.

## Steps

1. Identify actor group and record:
   - **Professional** — [R1](https://doi.org/10.5281/zenodo.20709433)
   - **Amateur** — [R2](https://doi.org/10.5281/zenodo.20709423)
   - **General (non-actor)** — [R3](https://doi.org/10.5281/zenodo.20721338)
2. Request access: read [data_use_agreement.md](../data_use_agreement.md), then [access_request.md](../access_request.md)
3. Download only the `*_subj<N>.zip` files you need.
4. Unzip — each archive has `<subject_id>/*.mp4` at the zip root.
5. With your download, read **`README_video.md`** and **`video_inventory_<group>.csv`** for naming and released trial counts.

## Align with neuro/behavior

Download [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) and use `data/subject_data_inventory.csv` (column definitions in [data/README.md](../../data/README.md#trial-inventory)).

## Citation

Cite the open data DOI: https://doi.org/10.5281/zenodo.20707429

## Other goals

- Neuro only → [neuro_only.md](neuro_only.md)
- Full reproduction → [full_reproduction.md](full_reproduction.md)
