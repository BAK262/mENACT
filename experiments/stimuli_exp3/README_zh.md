# Performance enactment 脚本（受限）

[![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md) [![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)

---

**开放包**不含 enactment 转录文本。请申请并合并受限脚本包 — 见 [docs/downloads_zh.md](../../docs/downloads_zh.md) 与 [docs/access_request_zh.md](../../docs/access_request_zh.md)。

合并后本目录包含：

- `script_matching.csv` — 每名被试各情绪脚本的来源被试 ID
- `privacy_policy.txt` — 转录文本匿名化规则
- `{1..53}/` — `*-transcript.txt`（发布集共 428 条脚本）

## 文件命名

```text
{status_}{index}_exp{N}_{timestamp}_{emotion}mp4-transcript.txt
```

| 前缀 | 含义 |
|------|------|
| 无 | 自动转录 |
| `f-` | 人工校对 |
| `s-` | 标记含隐私敏感内容 |
| `f-s-` | 已校对且含隐私标记 |

示例：`f-04_exp2_20240120201641_fearmp4-transcript.txt`

逐被试发布范围见 [data/README_zh.md#试次台账](../../data/README_zh.md#试次台账)。

---

**更新**：2026-06-17
