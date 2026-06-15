<p align="center">
  <img src="docs/assets/menact_logo.png" alt="mENACT logo" width="280">
</p>

# mENACT (Multimodal Expression-centered Neurophysiological Affective Computing Tasks)

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red.svg)](README_zh.md)

Version 1.0.0 · N=53 · fNIRS + three expression tasks (Passive Perception, Spontaneous Production, Deliberate Performance)

**Software and reproduction hub** — workflows, environments, validation code, and assembly guides.  
**Open data (cite in papers):** [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) · [downloads.md](docs/downloads.md)

## Choose your path

| Goal | Start here |
|------|------------|
| **Open neuro + behavior only** | [Zenodo R0](https://doi.org/10.5281/zenodo.20707429) · [docs/use_cases/neuro_only.md](docs/use_cases/neuro_only.md) |
| **Participant video only** | [docs/use_cases/video_only.md](docs/use_cases/video_only.md) |
| **Full reproduction** | [docs/use_cases/full_reproduction.md](docs/use_cases/full_reproduction.md) |

## Quick start (full dataset root)

See **[docs/assembly.md](docs/assembly.md)** for the complete step-by-step guide.

```powershell
# After unzipping R0 into your dataset root:
powershell -ExecutionPolicy Bypass -File scripts/assemble_dataset.ps1 -Root .
```

## More information

- [docs/downloads.md](docs/downloads.md) ([中文](docs/downloads_zh.md)) — product catalog and DOIs
- [data/README.md](data/README.md) ([中文](data/README_zh.md)) — data layout (also in R0)
- [docs/reproduce.md](docs/reproduce.md) ([中文](docs/reproduce_zh.md)) — technical validation workflows

## Citation

Cite **Zenodo R0**: https://doi.org/10.5281/zenodo.20707429 · See [CITATION.cff](CITATION.cff).

## License

Data (Zenodo): CC BY-NC 4.0 · Code (GitHub): MIT · Restricted recordings/scripts: [data use agreement](docs/data_use_agreement.md) ([中文](docs/data_use_agreement_zh.md))
