# Data Overview and Format Guide

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md)

---

Experimental data for **mENACT v1.0.0** (N=53): three fNIRS tasks (Perception, Production, Performance) plus individual-level trait measures (questionnaires and ECS).

## Task names

Three within-subject tasks operationalize a hierarchy of emotional engagement (manuscript labels in parentheses):

| Manuscript task | Experiment / files | Description |
|-----------------|-------------------|-------------|
| **Passive Perception** (Perception) | Experiment 1, `exp1.*` | Video clip viewing |
| **Spontaneous Production** (Production) | Experiment 2, `exp2.*` | Personal narrative |
| **Deliberate Performance** (Performance) | Experiment 3, `exp3.*` | Scripted enactment |

ECS (emotion concept similarity) is a separate PsychoPy task; output is `trait_ecs.csv` per participant (no fNIRS).

## Directory structure (this release)

```text
data/
├── README.md / README_zh.md
├── subject_data_inventory.csv
├── Brodmann_ROI.csv
├── all_raw/
│   ├── README.md / README_zh.md
│   ├── subject_info.csv
│   └── {1..53}/
└── fnirs_signals/
    └── AEPO_001filt02/
```

## Cohort summary

- **N**: 53 native Chinese speakers (professional / amateur / general [non-actor] groups)
- **Collection**: January–April 2024
- **Screening**: BDI-II ≤ 13; see `all_raw/subject_info.csv` for demographics and `depressScore`

## Trial inventory {#trial-inventory}

[`subject_data_inventory.csv`](subject_data_inventory.csv) — one row per participant (`id` matches `all_raw/subject_info.csv`), plus a `TOTAL` row. Technical validation uses **released trials only** (counts in this table).

### Columns

| Column | Unit counted |
|--------|----------------|
| `id` | Participant ID (1–53) |
| `group` | `professional`, `amateur`, or `general` (non-actor in the manuscript) |
| `perception_fnirs` | Perception trials in released preprocessed fNIRS (`AEPO_001filt02`) |
| `perception_rating` | Perception trials with viewing completion ≥ 95% in `exp1_*_rating.csv` |
| `production_fnirs` | Production trials in released preprocessed fNIRS |
| `production_mp4` | Production expression MP4 files (`exp2_*.mp4`) in the release tree |
| `production_rating` | Production narration trials (`tellCompletedPerc` > 1) across all `exp2_*_rating.csv` sessions |
| `performance_fnirs` | Performance trials in released preprocessed fNIRS |
| `performance_mp4` | Performance expression MP4 files (`exp3_*.mp4`) in the release tree |
| `performance_rating` | Performance enactment trials (`actCompletedPerc` > 1) in `exp3_*_rating.csv` |
| `notes` | Per-participant gaps (missing, skipped, failed, not distributed, repeated session) |

### Consent-scoped distribution (MP4 and enactment scripts)

Enactment scripts are narrative-derived transcripts in `experiments/stimuli_exp3/` (restricted package; 428 files total). Product catalog: [docs/downloads.md](https://github.com/BAK262/mENACT/blob/main/docs/downloads.md) ([中文](https://github.com/BAK262/mENACT/blob/main/docs/downloads_zh.md)).

| ID | MP4 | Enactment scripts |
|----|-----|-------------------|
| 2, 10, 15, 33, 39 | Production and Performance MP4s not distributed | Distributed |
| 17 | Production MP4s not distributed | Not distributed |
| 7 | Production disgust MP4 not distributed | Disgust script not distributed |
| 20 | Production and Performance MP4s not distributed | Distributed |

Per-participant detail: `notes` column in the CSV.

## Further reading

- [Raw data guide](all_raw/README.md) — file naming, formats, special cases
- [fNIRS preprocessed data](fnirs_signals/README.md) — AEPO pipeline outputs
- [Reproduce validation](https://github.com/BAK262/mENACT/blob/main/docs/reproduce.md) ([中文](https://github.com/BAK262/mENACT/blob/main/docs/reproduce_zh.md)) — analysis scripts
- [Neuro-only use case](https://github.com/BAK262/mENACT/blob/main/docs/use_cases/neuro_only.md) ([中文](https://github.com/BAK262/mENACT/blob/main/docs/use_cases/neuro_only_zh.md)) · [Download packages](https://github.com/BAK262/mENACT/blob/main/docs/downloads.md) ([中文](https://github.com/BAK262/mENACT/blob/main/docs/downloads_zh.md))

---

**Principal Investigator**: Ming Li · <liming16@tsinghua.org.cn>
**Last Updated**: 2026-06-18
