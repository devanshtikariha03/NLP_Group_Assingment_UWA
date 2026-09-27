# CITS4012 Assignment 2 — submission handoff

The completed provisional submission files are:

- `CITS4012_GROUP_ID.pdf`
- `CITS4012_GROUP_ID.ipynb`

Before LMS submission, replace `GROUP_ID` in both filenames, in the report author field, and in the notebook title/metadata. Replace the Team Contributions placeholder in `report/report.tex`, then rebuild and copy the PDF.

## Rebuild commands

From `report/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

From the project root, regenerate the notebook after any implementation/result changes:

```powershell
python build_notebook.py
```

## Experiment evidence

The complete local evidence is in `results/`: epoch histories, per-example test predictions, aggregate metrics, attention data/figures, run configuration, dataset statistics, and the seed-13 selected checkpoint.

The report uses the official ACL `final` style. It has five PDF pages total; assessed content ends on page 4, and Team Contributions plus references occupy page 5. The assignment's referenced code-template file was not present among the supplied PDF and dataset archives, so the notebook uses a complete conventional Colab structure and records that fact.
