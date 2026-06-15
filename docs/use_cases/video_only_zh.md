# 仅被试视频

[![English](https://img.shields.io/badge/Language-English-blue.svg)](video_only.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](video_only_zh.md)

若仅做 **MP4** 分析（如行为或视觉建模），一般**不需要** GitHub 或 Zenodo R0，除非要与 fNIRS/行为试次对齐。

## 步骤

1. 确认演员组与记录：
   - **专业** — [R1](https://doi.org/10.5281/zenodo.20709433)
   - **业余** — [R2](https://doi.org/10.5281/zenodo.20709423)
   - **普通（非演员）** — [R3](https://doi.org/10.5281/zenodo.20721338)
2. 申请访问：阅读 [data_use_agreement_zh.md](../data_use_agreement_zh.md)，再 [access_request_zh.md](../access_request_zh.md)
3. 仅下载所需的 `*_subj<N>.zip`。
4. 解压 — 每个压缩包根目录为 `<subject_id>/*.mp4`。
5. 随包阅读 **`README_video.md`** 与 **`video_inventory_<group>.csv`**（命名与已发布试次数）。

## 与神经/行为对齐

下载 [Zenodo R0](https://doi.org/10.5281/zenodo.20707429)，使用 `data/subject_data_inventory.csv`（列定义见 [data/README_zh.md](../../data/README_zh.md#试次台账)）。

## 引用

引用开放数据 DOI：https://doi.org/10.5281/zenodo.20707429

## 其他目标

- 仅神经 → [neuro_only_zh.md](neuro_only_zh.md)
- 完整复现 → [full_reproduction_zh.md](full_reproduction_zh.md)
