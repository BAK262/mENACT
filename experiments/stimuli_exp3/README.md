# Performance enactment scripts (restricted)

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md)

---

The **open package** does not include enactment transcripts. Request access and merge the restricted scripts package — see [docs/downloads.md](../../docs/downloads.md) ([中文](../../docs/downloads_zh.md)) and [docs/access_request.md](../../docs/access_request.md) ([中文](../../docs/access_request_zh.md)).

After merge, this directory contains:

- `script_matching.csv` — source participant ID per emotion script for each recipient
- `privacy_policy.txt` — anonymization rules applied to transcripts
- `{1..53}/` — `*-transcript.txt` files (428 scripts total in the release)

## File naming

```text
{status_}{index}_exp{N}_{timestamp}_{emotion}mp4-transcript.txt
```

| Prefix | Meaning |
|--------|---------|
| (none) | Automatic transcription |
| `f-` | Manually revised |
| `s-` | Privacy-sensitive content flagged |
| `f-s-` | Revised and privacy-flagged |

Example: `f-04_exp2_20240120201641_fearmp4-transcript.txt`

Distribution scope per participant: [data/README.md#trial-inventory](../../data/README.md#trial-inventory).

---

**Last Updated**: 2026-06-17
