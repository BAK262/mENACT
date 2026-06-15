# 实验范式

[![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md) [![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)

---

mENACT 的 PsychoPy 协议：受试者内 fNIRS，涵盖 **Passive Perception**（视频观看）、**Spontaneous Production**（个人叙述）、**Deliberate Performance**（脚本 enactment）。数据布局与试次计数见 [data/README_zh.md](../data/README_zh.md)。

## 任务与脚本

| 论文任务 | 脚本 / 前缀 | 说明 |
|---------|------------|------|
| **Passive Perception**（Perception / 感知） | `main_exp1.py`, `exp1.*` | 视频观看 |
| **Spontaneous Production**（Production / 产出） | `main_exp2.py`, `exp2.*` | 个人叙述 |
| **Deliberate Performance**（Performance / 表演） | `main_exp3.py`, `exp3.*` | 脚本 enactment |
| **ECS**（特质） | `main_ecs.py`, `trait_ecs.csv` | 情绪概念相似性（无 fNIRS） |

九类情绪：tenderness, joy, inspiration, amusement, neutral, sadness, fear, disgust, anger。

## 环境

```bash
conda env create -f environments/ee_experiments.yml
```

PsychoPy 相机与 MoviePy 所需补丁见 `utils_exp.py`（安装后按注释修改）。

## 输出文件（位于 `data/all_raw/{id}/`）

详见 [data/all_raw/README_zh.md](../data/all_raw/README_zh.md)（[English](../data/all_raw/README.md)）。

| 任务 | 主要输出 |
|------|----------|
| Perception | `exp1_{timestamp}_rating.csv`, `exp1_{timestamp}_math.csv`, `exp1.nirs` |
| Production | `exp2_{timestamp}_rating.csv`, `exp2_{timestamp}_math.csv`, `exp2_{timestamp}_{emotion}.mp4`, `exp2.nirs` |
| Performance | `exp3_{timestamp}_rating.csv`, `exp3_{timestamp}_math.csv`, `exp3_{timestamp}_{emotion}.mp4`, `exp3.nirs` |
| ECS | `trait_ecs.csv` |

fNIRS 触发码：`task_event.xlsx`。Perception 刺激：`stimuli_exp1/`（28 片段，`stimuli_info.xlsx`）。

## Performance 脚本（`stimuli_exp3/`）

Performance 的同伴衍生 enactment 文本在**受限脚本包**中。开放包仅含占位说明 — 见 [stimuli_exp3/README_zh.md](stimuli_exp3/README_zh.md) 与 [docs/downloads_zh.md](../docs/downloads_zh.md)。合并后脚本分配见 `stimuli_exp3/script_matching.csv`。

MP4 与脚本的同意范围见 [data/README_zh.md#试次台账](../data/README_zh.md#试次台账)。

## 目录结构

```text
experiments/
├── main_exp1.py, main_exp2.py, main_exp3.py, main_ecs.py
├── utils_exp.py
├── task_event.xlsx
├── stimuli_exp1/
└── stimuli_exp3/
```

## 伦理

伦理批件 THU202312（清华大学）。引用见根目录 [CITATION.cff](../CITATION.cff)。

---

**负责人**：黎明 · <liming16@tsinghua.org.cn>  
**更新**：2026-06-15
