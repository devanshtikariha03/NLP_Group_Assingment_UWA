# CITS4012 NLP Group Assignment — PIQA

This repository contains a complete, reproducible investigation of physical commonsense question answering on PIQA for CITS4012 Assignment 2.

The main model is **GACA-BiGRU**: a from-scratch shared bidirectional GRU with goal-conditioned additive attention, explicit goal–candidate interaction features, and a swap-equivariant contrastive candidate scorer. Across three seeds it achieves **60.65 ± 0.33% test accuracy** and **60.64 ± 0.33% macro-F1**.

## Main files

- `CITS4012_GROUP_ID.ipynb` — self-contained Colab implementation with saved logs, results, ablations, and attention visualisations.
- `CITS4012_GROUP_ID.pdf` — compiled ACL-format report; replace the group placeholder before submission.
- `piqa_experiments.py` — command-line implementation used to run the recorded experiments.
- `report/` — LaTeX source, official ACL style files, bibliography, figures, and compiled PDF.
- `results/` — configurations, epoch histories, predictions, aggregate metrics, attention data, and the selected seed-13 checkpoint.
- `README_SUBMISSION.md` — final submission and rebuild checklist.

## Reproduce experiments

With Python 3.10+ and the supplied PIQA archive available in the repository root:

```powershell
python piqa_experiments.py --root . --output-dir results --epochs 6 --patience 2 --batch-size 256
```

The notebook provides the recommended Google Colab workflow and automatically extracts the PIQA archive.

## Build the report

```powershell
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

Before submission, replace every `GROUP_ID` placeholder and complete the Team Contributions section.
