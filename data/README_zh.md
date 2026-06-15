# 数据概览与格式说明

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md)

---

**mENACT v1.0.0** 实验数据（N=53）：三个 fNIRS 任务（Perception / Production / Performance）及个体水平特质测量（问卷与 ECS）。

## 任务名称

受试者内三项任务对应手稿中的三级情绪参与层次（括号内为简称）：

| 论文任务 | 实验 / 文件 | 说明 |
|---------|------------|------|
| **Passive Perception**（Perception / 感知） | 实验 1，`exp1.*` | 视频观看 |
| **Spontaneous Production**（Production / 产出） | 实验 2，`exp2.*` | 个人叙述 |
| **Deliberate Performance**（Performance / 表演） | 实验 3，`exp3.*` | 脚本 enactment |

ECS（情绪概念相似性）为独立 PsychoPy 任务，输出各被试 `trait_ecs.csv`（无 fNIRS）。

## 目录结构（本发布包）

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

## 样本概要

- **N=53**，专业 / 业余 / 普通（`general`，对应手稿 non-actor）三组
- **采集**：2024 年 1–4 月
- **筛查**：BDI-II ≤ 13；人口学见 `all_raw/subject_info.csv`

## 试次台账 {#试次台账}

[`subject_data_inventory.csv`](subject_data_inventory.csv) — 每名被试一行（`id` 与 `all_raw/subject_info.csv` 一致），末行为 `TOTAL`。技术验证仅使用**已发布试次**（本表计数）。

### 列定义

| 列 | 计数对象 |
|----|----------|
| `id` | 被试编号（1–53） |
| `group` | `professional` / `amateur` / `general`（手稿 non-actor） |
| `perception_fnirs` | 发布预处理 fNIRS 中的 Perception 试次 |
| `perception_rating` | `exp1_*_rating.csv` 中观看完成度 ≥ 95% 的试次 |
| `production_fnirs` | 发布预处理 fNIRS 中的 Production 试次 |
| `production_mp4` | 发布树中的 Production MP4（`exp2_*.mp4`） |
| `production_rating` | 全部 `exp2_*_rating.csv` 中 `tellCompletedPerc` > 1 的叙述试次 |
| `performance_fnirs` | 发布预处理 fNIRS 中的 Performance 试次 |
| `performance_mp4` | 发布树中的 Performance MP4（`exp3_*.mp4`） |
| `performance_rating` | `exp3_*_rating.csv` 中 `actCompletedPerc` > 1 的 enactment 试次 |
| `notes` | 该被试的缺失、跳过、失败、未发布、重测等说明 |

### 同意范围内的发布（MP4 与 enactment scripts）

语义衍生指 self narrative 转录整理得到的 enact scripts（`experiments/stimuli_exp3/`，受限包共 428 条）。产品目录见 [docs/downloads_zh.md](https://github.com/BAK262/mENACT/blob/main/docs/downloads_zh.md)。

| ID | MP4 | enactment scripts |
|----|-----|-------------------|
| 2, 10, 15, 33, 39 | Production / Performance 录像未发布 | 已发布 |
| 17 | Production 录像未发布 | 未发布 |
| 7 | Production disgust 录像未发布 | disgust 脚本未发布 |
| 20 | Production / Performance 录像未发布 | 已发布 |

逐被试说明见 CSV 的 `notes` 列。

## 延伸阅读

- [原始数据说明](all_raw/README_zh.md) — 文件命名、格式、特例
- [fNIRS 预处理数据](fnirs_signals/README_zh.md) — AEPO 管线输出
- [复现验证](https://github.com/BAK262/mENACT/blob/main/docs/reproduce_zh.md) — 分析脚本
- [仅开放数据](https://github.com/BAK262/mENACT/blob/main/docs/use_cases/neuro_only_zh.md) · [下载包](https://github.com/BAK262/mENACT/blob/main/docs/downloads_zh.md)

---

**负责人**：黎明 · <liming16@tsinghua.org.cn>
**更新**：2026-06-18
