# Experimental Paradigms

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md)

---

PsychoPy protocols for mENACT: within-subject fNIRS across **Passive Perception** (video viewing), **Spontaneous Production** (personal narrative), and **Deliberate Performance** (scripted enactment). Data layout and trial counts: [data/README.md](../data/README.md).

## Tasks and scripts

| Manuscript task | Script / prefix | Description |
|-----------------|-----------------|-------------|
| **Passive Perception** (Perception) | `main_exp1.py`, `exp1.*` | Video clip viewing |
| **Spontaneous Production** (Production) | `main_exp2.py`, `exp2.*` | Personal narrative |
| **Deliberate Performance** (Performance) | `main_exp3.py`, `exp3.*` | Scripted enactment |
| **ECS** (trait) | `main_ecs.py`, `trait_ecs.csv` | Emotion concept similarity (no fNIRS) |

Nine emotions across tasks: tenderness, joy, inspiration, amusement, neutral, sadness, fear, disgust, anger.

## Environment

```bash
conda env create -f environments/ee_experiments.yml
```

Package patches required for PsychoPy camera and MoviePy are documented in `utils_exp.py` (apply after install).

## Output files (under `data/all_raw/{id}/`)

Detail: [data/all_raw/README.md](../data/all_raw/README.md) ([中文](../data/all_raw/README_zh.md)).

| Task | Key outputs |
|------|-------------|
| Perception | `exp1_{timestamp}_rating.csv`, `exp1_{timestamp}_math.csv`, `exp1.nirs` |
| Production | `exp2_{timestamp}_rating.csv`, `exp2_{timestamp}_math.csv`, `exp2_{timestamp}_{emotion}.mp4`, `exp2.nirs` |
| Performance | `exp3_{timestamp}_rating.csv`, `exp3_{timestamp}_math.csv`, `exp3_{timestamp}_{emotion}.mp4`, `exp3.nirs` |
| ECS | `trait_ecs.csv` |

fNIRS trigger codes: `task_event.xlsx`. Perception stimuli: `stimuli_exp1/` (28 clips, `stimuli_info.xlsx`).

## Performance scripts (`stimuli_exp3/`)

Peer-derived enactment transcripts for Performance are in the **restricted scripts** package. Open tree: placeholder README only — see [stimuli_exp3/README.md](stimuli_exp3/README.md) and [docs/downloads.md](../docs/downloads.md) ([中文](../docs/downloads_zh.md)). Script assignment: `stimuli_exp3/script_matching.csv` (after merge).

Consent-scoped MP4 and script distribution: [data/README.md#trial-inventory](../data/README.md#trial-inventory).

## Directory layout

```text
experiments/
├── main_exp1.py, main_exp2.py, main_exp3.py, main_ecs.py
├── utils_exp.py
├── task_event.xlsx
├── stimuli_exp1/
└── stimuli_exp3/
```

## Ethics

IRB THU202312 (Tsinghua University). Cite via root [CITATION.cff](../CITATION.cff).

---

**Principal Investigator**: Ming Li · <liming16@tsinghua.org.cn>  
**Last Updated**: 2026-06-15
