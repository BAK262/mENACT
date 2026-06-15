# 受限数据申请（音视频 + 脚本）

[![English](https://img.shields.io/badge/Language-English-blue.svg)](access_request.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](access_request_zh.md)

被试视频（R1–R3）与 enactment 脚本（R4）**不在**开放 R0 包中。

## 申请步骤

1. **阅读 [数据使用协议](data_use_agreement_zh.md)**。
2. 打开 [申请表](https://forms.gle/UihkbfYnaaZjD8Cp7)（表单界面为英文）。
3. 在 **mENACT data use agreement (select all)** 下，阅读协议后**勾选全部**选项。
4. 填写 **Full name**、**Institutional email**（用于接收下载链接，请使用大学/研究所邮箱）、**Affiliation**、**PI or supervisor**、**Research purpose**。
5. **Ethics / IRB status (or N/A)**：若*计划用途*需在本机构伦理审查，填写批件号与机构；若仅分析既有去标识数据、无新的人体研究，写 **N/A** 并简要说明。此处**不是**请贵方 IRB 批准原始 mENACT 研究。
6. **Requested package**（**三选一**）：**Restricted AV**（全部演员组）、**Restricted scripts**、**Both**。Restricted AV 含专业/业余/非演员视频；表单上不能单独选择某一演员组。
7. 提交后，下载链接将发送至所填机构邮箱（请检查垃圾邮件）。

## 下载之后

| 包 | 下一步 |
|----|--------|
| **仅 AV** | [use_cases/video_only_zh.md](use_cases/video_only_zh.md) |
| **仅脚本** | 解压 R4；可选按 [assembly_zh.md](assembly_zh.md) 第 3 步合并 |
| **完整数据集** | [assembly_zh.md](assembly_zh.md) |

构建数据集根目录时的合并命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/restore_restricted.ps1 -Root . -Parts ".\downloads\menact_v1.0.0_restricted_*.zip"
```

产品 DOI：[downloads_zh.md](downloads_zh.md)。
