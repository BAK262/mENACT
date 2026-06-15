# 复现技术验证

[![English](https://img.shields.io/badge/Language-English-blue.svg)](reproduce.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](reproduce_zh.md)

**前提：** 完整的[数据集根目录](assembly_zh.md)（R0 + GitHub 标签 `v1.0.0`；重跑语音验证前须合并受限 MP4）。组装：[assembly_zh.md](assembly_zh.md)。

在**数据集根目录**（含 `VERSION` 的目录）运行。

所有工作流支持 `--quick`（小样本冒烟，默认）或 `--full`（N=53）。

## 特质（Traits）

```powershell
conda activate ee_validation_speech
python code/validation_traits.py --quick
```

## 自陈（Self-report）

```powershell
python code/validation_selfreport.py --quick
```

## 语音（完整重跑需已合并受限 AV）

**重跑**工作流前须先合并受限 MP4，然后：

```powershell
conda env create -f environments/ee_validation_speech.yml
python code/validation_speech.py --full
```

## fNIRS 质量

```powershell
python code/validation_fnirs_quality.py --quick
```

## fNIRS 解码

```powershell
conda env create -f environments/ee_validation_fnirs_decoding.yml
python code/validation_fnirs_decoding.py --full
```

`results/validation_*/` 下预置输出仅为**手稿引用产物**（表格、PDF 图及 `.gitignore` 所列统计 CSV）。完整重跑会在本地写入更多文件。所有验证计数仅使用**已发布试次**（见 [data/README_zh.md](../data/README_zh.md#试次台账) 与 `data/subject_data_inventory.csv`）。

Homer3、FieldTrip、OpenSMILE 安装见 [install_zh.md](install_zh.md)。
