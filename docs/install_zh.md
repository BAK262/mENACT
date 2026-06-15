# 外部依赖（fNIRS 预处理与语音特征）

[![English](https://img.shields.io/badge/Language-English-blue.svg)](install.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](install_zh.md)

## fNIRS（MATLAB）

- MATLAB R2023b（或兼容版本）
- [Homer3](https://github.com/BUNPC/Homer3) v1.80.2
- [FieldTrip](https://www.fieldtriptoolbox.org/) 20240110

发布包在 `code/utils/fnirs_preprocess/` 提供预处理辅助脚本，但**不包含**完整 Homer3 / FieldTrip 树（体积过大）。请按 `code/fnirs_preprocess.m` 期望路径在本地安装。

### 1. 安装第三方工具箱

在**数据集根目录**下：

```powershell
cd code\utils

# Homer3 v1.80.2
git clone --branch v1.80.2 --depth 1 https://github.com/BUNPC/Homer3.git Homer3-1.80.2

# FieldTrip 20240110 — 从 fieldtriptoolbox.org 下载发布 zip，
# 解压并重命名文件夹为 fieldtrip-20240110。
```

期望布局：

```text
code/utils/
├── Homer3-1.80.2/
├── fieldtrip-20240110/
├── fnirs_preprocess/
└── homer3_patches/
```

### 2. 应用 Homer3 补丁

预处理前须修补两个文件（亦见 `code/fnirs_preprocess.m` 文件头）：

| 补丁 | 目的 |
|------|------|
| `hdf5write_safe.m` 第 98 行：`H5T_NATIVE_ULONG` → `H5T_NATIVE_INT` | macOS 上 SNIRF 保存（[Homer3#181](https://github.com/BUNPC/Homer3/issues/181)） |
| `hmrR_PruneChannels.m` | 任一波长未通过质量检查时两波长均修剪 |

预补丁副本位于 `code/utils/homer3_patches/`。克隆 Homer3 后执行：

```powershell
.\scripts\apply_homer3_patches.ps1
```

或手动复制：

```powershell
Copy-Item code\utils\homer3_patches\FuncRegistry\UserFunctions\hmrR_PruneChannels.m `
  code\utils\Homer3-1.80.2\FuncRegistry\UserFunctions\ -Force
Copy-Item code\utils\homer3_patches\DataTree\AcquiredData\DataFiles\Hdf5\hdf5write_safe.m `
  code\utils\Homer3-1.80.2\DataTree\AcquiredData\DataFiles\Hdf5\ -Force
```

### 3. 运行预处理

在 MATLAB 中 **cd 到数据集根目录**（含 `VERSION` 的目录），然后：

```matlab
run('code/fnirs_preprocess.m')
```

脚本会添加路径、运行 Homer3 `setpaths` 与 FieldTrip `ft_defaults`。原始 `.nirs` 须在 `data/all_raw/{subject}/`，并有对应 `*_rating.csv` 试次日志。

输出目录：`data/fnirs_signals/AEPO_001filt02/`（带通 0.01–0.2 Hz）。

## 语音 eGeMAPS（可选）

- [OpenSMILE](https://github.com/audeering/opensmile) 3.0 — Windows x64 构建，或通过 conda 环境 `ee_validation_speech` 使用 `python-opensmile`

HuBERT 特征需 GPU 及 `environments/ee_validation_speech.yml` 所列包。

## Conda 环境

| 文件 | 用途 |
|------|------|
| `environments/ee_experiments.yml` | PsychoPy 实验复现 |
| `environments/ee_validation_speech.yml` | 语音验证 |
| `environments/ee_validation_fnirs_decoding.yml` | fNIRS MIL 解码 |

```powershell
conda env create -f environments/ee_validation_speech.yml
```
