# 下载产品目录

[![English](https://img.shields.io/badge/Language-English-blue.svg)](downloads.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](downloads_zh.md)

**版本 1.0.0** · 主引用 DOI：[10.5281/zenodo.20707429](https://doi.org/10.5281/zenodo.20707429)

每一行是一个**独立产品**。完整论文复现需组合多个产品；见 [assembly_zh.md](assembly_zh.md)。

| 产品 | 获取方式 | 可独立用于 | 内容（摘要） |
|------|----------|------------|--------------|
| **R0 — 开放数据** | [Zenodo](https://doi.org/10.5281/zenodo.20707429)（公开） | fNIRS + 行为 CSV 分析 | `VERSION`、`LICENSE`、`data/`（无 MP4）、`QUICKSTART_neuro.md`、`ASSEMBLY.md` |
| **R1 — 受限 AV（专业演员）** | [Zenodo](https://doi.org/10.5281/zenodo.20709433)（受限） | 按被试 MP4 | 每被试一 zip：`<id>/*.mp4` + 存证 `README_video.md` |
| **R2 — 受限 AV（业余演员）** | [Zenodo](https://doi.org/10.5281/zenodo.20709423)（受限） | 按被试 MP4 | 布局同 R1 |
| **R3 — 受限 AV（普通被试）** | [Zenodo](https://doi.org/10.5281/zenodo.20721338)（受限） | 按被试 MP4 | 布局同 R1 |
| **R4 — 受限脚本** | [Zenodo](https://doi.org/10.5281/zenodo.20709427)（受限） | enactment 转录 | `stimuli_exp3/*/*.txt` + `README_scripts.md` |
| **GitHub — 工程树** | [BAK262/mENACT](https://github.com/BAK262/mENACT) 标签 `v1.0.0` | 代码、文档、验证 | `code/`、`experiments/`（无 txt）、`docs/`、`scripts/`、`results/` 子集 |

社区：[Zenodo mENACT](https://zenodo.org/communities/menact)

## 选择路径

| 目标 | 指南 |
|------|------|
| 仅开放神经与行为 | [use_cases/neuro_only_zh.md](use_cases/neuro_only_zh.md) |
| 仅被试视频 | [use_cases/video_only_zh.md](use_cases/video_only_zh.md) |
| 完整复现 | [use_cases/full_reproduction_zh.md](use_cases/full_reproduction_zh.md) |

受限包：先读 [数据使用协议](data_use_agreement_zh.md)，再 [access_request_zh.md](access_request_zh.md)。
