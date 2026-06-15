<p align="center">
  <img src="docs/assets/menact_logo.png" alt="mENACT logo" width="280">
</p>

# mENACT（Multimodal Expression-centered Neurophysiological Affective Computing Tasks）

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md)

版本 1.0.0 · N=53 · fNIRS + 三项表达任务（Passive Perception、Spontaneous Production、Deliberate Performance）

**软件与复现中心** — 工作流、环境、验证代码与组装指南。  
**开放数据（论文引用）：** [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) · [downloads_zh.md](docs/downloads_zh.md)

## 选择路径

| 目标 | 入口 |
|------|------|
| **仅开放神经与行为数据** | [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) · [docs/use_cases/neuro_only_zh.md](docs/use_cases/neuro_only_zh.md) |
| **仅被试视频** | [docs/use_cases/video_only_zh.md](docs/use_cases/video_only_zh.md) |
| **完整论文复现** | [docs/use_cases/full_reproduction_zh.md](docs/use_cases/full_reproduction_zh.md) |

## 快速开始（完整数据集根目录）

完整步骤见 **[docs/assembly_zh.md](docs/assembly_zh.md)**。

```powershell
# 将 R0 解压到数据集根目录后：
powershell -ExecutionPolicy Bypass -File scripts/assemble_dataset.ps1 -Root .
```

## 更多信息

- [docs/downloads_zh.md](docs/downloads_zh.md) — 产品目录与 DOI
- [data/README_zh.md](data/README_zh.md) — 数据布局（亦含于 R0）
- [docs/reproduce_zh.md](docs/reproduce_zh.md) — 技术验证工作流

## 引用

引用 **Zenodo R0**：https://doi.org/10.5281/zenodo.20707429 · 见 [CITATION.cff](CITATION.cff)。

## 许可

数据（Zenodo）：CC BY-NC 4.0 · 代码（GitHub）：MIT · 受限录像/脚本：[数据使用协议](docs/data_use_agreement_zh.md)
