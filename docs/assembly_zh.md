# 组装完整数据集根目录

[![English](https://img.shields.io/badge/Language-English-blue.svg)](assembly.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](assembly_zh.md)

**数据集根目录**指包含 `VERSION`、`data/`、`code/`（及相关工程树）的单一文件夹。本文是**唯一**载有完整组装步骤的文档。

**前提：** 需运行 [reproduce_zh.md](reproduce_zh.md) 中的论文工作流。仅神经或仅视频用途见 [neuro_only_zh.md](use_cases/neuro_only_zh.md)、[video_only_zh.md](use_cases/video_only_zh.md)。

## 步骤

### 1. 开放数据（Zenodo R0）

下载 [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) 并解压到数据集根目录（将包含 `VERSION` 与 `data/` 的文件夹）。

### 2. 工程树（GitHub）

在数据集根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/assemble_dataset.ps1 -Root .
```

或手动克隆标签 `v1.0.0`，将 `code/`、`experiments/`、`environments/`、`docs/`、`scripts/`、`results/` 移入根目录。

仓库：https://github.com/BAK262/mENACT

### 3. 可选 — 受限 MP4 与脚本

1. 阅读 [data_use_agreement_zh.md](data_use_agreement_zh.md)，经 [access_request_zh.md](access_request_zh.md) 申请。
2. 使用邮件中的链接下载 zip。
3. 合并：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/restore_restricted.ps1 -Root . -Parts ".\downloads\menact_v1.0.0_restricted_*.zip"
```

### 4. 环境

```powershell
conda env create -f environments/ee_validation_speech.yml
```

详见 [install_zh.md](install_zh.md)。

### 5. 校验

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_layout.ps1 -Root . -Profile full
python code/validation_traits.py --quick
```

## 产品索引

[downloads_zh.md](downloads_zh.md)
