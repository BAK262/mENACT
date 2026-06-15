# Restricted data access (AV + scripts)

[![English](https://img.shields.io/badge/Language-English-blue.svg)](access_request.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](access_request_zh.md)

Restricted participant video (R1–R3) and enactment scripts (R4) are **not** in the open R0 package.

## How to request

1. **Read the [data use agreement](data_use_agreement.md)**.
2. Open the [request form](https://forms.gle/UihkbfYnaaZjD8Cp7).
3. On the form, under **mENACT data use agreement (select all)**, check **every** box after reading the agreement.
4. Fill in **Full name**, **Institutional email** (where we send download link(s) — use your university/institute address), **Affiliation**, **PI or supervisor**, and **Research purpose**.
5. **Ethics / IRB status (or N/A):** If your *planned use* needs ethics review at your institution, give approval ID and institution. If you only analyze existing de-identified data with no new human-subjects work, write **N/A** and briefly explain. This is not a request for your IRB to approve the original mENACT study.
6. **Requested package** (pick **one**): **Restricted AV** (all actor groups), **Restricted scripts**, or **Both**. Restricted AV includes professional, amateur, and non-actor videos; you cannot select a single actor group on the form.
7. Submit. You will receive **download link(s)** by email at the institutional address you entered (check spam).

## After download

| Package | Next step |
|---------|-----------|
| **AV only** | [use_cases/video_only.md](use_cases/video_only.md) |
| **Scripts only** | Unzip R4; optional merge via [assembly.md](assembly.md) step 3 |
| **Full dataset** | [assembly.md](assembly.md) |

Merge command (when building a dataset root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/restore_restricted.ps1 -Root . -Parts ".\downloads\menact_v1.0.0_restricted_*.zip"
```

Product DOIs: [downloads.md](downloads.md).
