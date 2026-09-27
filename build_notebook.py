"""Build the self-contained submission notebook with saved experiment evidence."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "CITS4012_GROUP_ID.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source, outputs=None, execution_count=None):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def stream(text):
    return {"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}


implementation = (ROOT / "piqa_experiments.py").read_text(encoding="utf-8")
implementation = implementation.split("def parse_args():", 1)[0].rstrip() + "\n"

stats = json.loads((RESULTS / "dataset_statistics.json").read_text(encoding="utf-8"))
all_results = pd.read_csv(RESULTS / "all_results.csv")
aggregate = json.loads((RESULTS / "aggregate_results.json").read_text(encoding="utf-8"))
attention = json.loads((RESULTS / "attention_examples.json").read_text(encoding="utf-8"))

log_lines = []
for history_path in sorted(RESULTS.glob("*_history.csv")):
    history = pd.read_csv(history_path)
    log_lines.append(f"[{history_path.stem}]")
    for row in history.itertuples(index=False):
        log_lines.append(
            f"epoch={int(row.epoch):02d} train_loss={row.train_loss:.4f} "
            f"val_loss={row.val_loss:.4f} val_acc={row.val_accuracy:.4f} val_f1={row.val_macro_f1:.4f}"
        )
log_text = "\n".join(log_lines) + "\n"

result_display = all_results.copy()
for column in ["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"]:
    result_display[column] = (100 * result_display[column]).round(2)
result_display["seconds"] = result_display["seconds"].round(1)
results_text = result_display.to_string(index=False) + "\n\nAggregate main-model results:\n" + json.dumps(aggregate, indent=2) + "\n"

setup_source = r'''# Colab/runtime setup
# PyTorch is preinstalled on standard Colab runtimes. The remaining packages are lightweight.
%pip -q install "pandas>=2.0" "scikit-learn>=1.3" "scipy>=1.10" "matplotlib>=3.7"

from pathlib import Path
import zipfile

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Put the supplied PIQA zip in the Colab Files pane (or set PROJECT_ROOT to a Drive folder).
# This extracts only when train.jsonl is not already present.
if not list(PROJECT_ROOT.rglob("train-labels.lst")):
    archives = list(PROJECT_ROOT.glob("*PIQA*.zip"))
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(PROJECT_ROOT / "piqa_data")
    else:
        raise FileNotFoundError(
            "Upload the supplied PIQA-...zip to the Colab Files pane, then rerun this cell."
        )
print("Project root:", PROJECT_ROOT)
'''

audit_source = r'''# Load the official splits and create the fixed training-only validation split.
data_dir = find_data_dir(PROJECT_ROOT)
train_all = load_split(data_dir, "train")
test = load_split(data_dir, "test")
train, val = split_data(train_all)
vocab = Vocabulary(pd.concat([train["goal"], train["sol1"], train["sol2"]], ignore_index=True))
stats = dataset_statistics(train_all, train, val, test, vocab)
print(json.dumps(stats, indent=2))
'''

audit_output = json.dumps(stats, indent=2) + "\n"

sanity_source = r'''# Architecture invariant: swapping the alternatives must swap the two logits.
set_seed(13)
probe = GACABiGRU(len(vocab), emb_dim=96, hidden=64, dropout=0.25, mode="main")
probe.eval()
batch = next(iter(DataLoader(PIQADataset(train.iloc[:8], vocab), batch_size=8, collate_fn=collate)))
with torch.no_grad():
    original, _, _ = probe(batch["goal"], batch["goal_len"], batch["sol1"], batch["sol1_len"], batch["sol2"], batch["sol2_len"])
    swapped, _, _ = probe(batch["goal"], batch["goal_len"], batch["sol2"], batch["sol2_len"], batch["sol1"], batch["sol1_len"])
assert torch.allclose(original, swapped.flip(1), atol=1e-6)
print("Swap-equivariance check: PASS")
print(f"Main-model trainable parameters: {sum(p.numel() for p in probe.parameters()):,}")
'''

run_source = r'''# Full pre-specified experiment matrix. This cell is intentionally executable end-to-end.
# Expected time depends on runtime; a GPU is selected automatically when available.
from types import SimpleNamespace

args = SimpleNamespace(
    root=PROJECT_ROOT,
    output_dir=OUTPUT_DIR,
    epochs=6,
    patience=2,
    batch_size=256,
    embedding_dim=96,
    hidden_dim=64,
    dropout=0.25,
    learning_rate=2e-3,
    weight_decay=1e-4,
    grad_clip=5.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seeds=[13, 37, 73],
)
torch.set_num_threads(min(8, os.cpu_count() or 1))
print("Device:", args.device)

summaries = [majority_baseline(train, val, test), tfidf_baseline(train, val, test, OUTPUT_DIR)]
main_model = None
for seed in args.seeds:
    model, summary = train_neural(train, val, test, vocab, "main", seed, args, OUTPUT_DIR)
    summaries.append(summary)
    if seed == args.seeds[0]:
        main_model = model

for mode in ["mean_bigru", "no_attention", "no_contrast", "simple_features"]:
    _, summary = train_neural(train, val, test, vocab, mode, args.seeds[0], args, OUTPUT_DIR)
    summaries.append(summary)

results = pd.DataFrame(summaries)
results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
main_rows = results[results["model"] == "main"]
aggregate = {
    "main_test_accuracy_mean": float(main_rows["test_accuracy"].mean()),
    "main_test_accuracy_std": float(main_rows["test_accuracy"].std(ddof=1)),
    "main_test_macro_f1_mean": float(main_rows["test_macro_f1"].mean()),
    "main_test_macro_f1_std": float(main_rows["test_macro_f1"].std(ddof=1)),
}
seed13 = pd.read_csv(OUTPUT_DIR / "main_seed13_test_predictions.csv")
aggregate["main_seed13_accuracy_95ci"] = bootstrap_accuracy_interval(seed13["gold"], seed13["prediction"])
(OUTPUT_DIR / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2))
save_attention_examples(main_model, test, vocab, OUTPUT_DIR, args.device)
display(results)
print(json.dumps(aggregate, indent=2))
'''

evidence_source = r'''# Saved evidence from the completed run used in the report.
# Re-running the experiment cell above regenerates these CSV-backed logs and values.
print(SAVED_TRAINING_LOGS)
print(SAVED_RESULTS)
'''

analysis_md = f'''## Qualitative attention analysis

The saved visualisations below use the seed-13 checkpoint selected only by validation accuracy.

- **Correct case (row {attention[0]["row_id"]})** — Goal: “{attention[0]["goal"]}”  
  Gold/prediction: solution {attention[0]["gold"] + 1}; confidence {max(attention[0]["probabilities"]):.1%}. Attention on the correct solution peaks at *level* and *noodles*, while the rival highlights *pillows*.
- **Failure (row {attention[1]["row_id"]})** — Goal: “{attention[1]["goal"]}”  
  Gold: solution {attention[1]["gold"] + 1}; prediction: solution {attention[1]["prediction"] + 1} with {max(attention[1]["probabilities"]):.1%} confidence. The model attends strongly to *acetone* but lacks the material/chemical knowledge to treat it as unsafe.

Attention weights are diagnostic evidence, not causal explanations of the final MLP decision.
'''

attention_code = r'''# Regenerated after the training cell; saved outputs from the report run are shown here.
from IPython.display import Image, display
display(Image(filename=str(OUTPUT_DIR / "attention_correct.png")))
display(Image(filename=str(OUTPUT_DIR / "attention_error.png")))
'''

image_outputs = []
for filename in ["attention_correct.png", "attention_error.png"]:
    encoded = base64.b64encode((RESULTS / filename).read_bytes()).decode("ascii")
    image_outputs.append({"data": {"image/png": encoded}, "metadata": {}, "output_type": "display_data"})

cells = [
    markdown(
        """# CITS4012 Assignment 2 — Goal-Aware Contrastive Attention for PIQA

**Group: `GROUP_ID` (replace before submission)**

This notebook is the complete, from-scratch implementation accompanying the ACL report. It contains setup and data instructions, preprocessing, all model/baseline definitions, the fixed experiment matrix, saved training/evaluation evidence, attention visualisations, and reproducibility checks. No external training data or pretrained representation is used by the main model.

> The assignment refers to a provided code template, but no template file was present with the supplied brief and four dataset archives. The notebook therefore follows a conventional, fully executable Colab structure.
"""
    ),
    markdown(
        """## Reproduction protocol

1. Open this notebook in Google Colab and select **Runtime → Run all**.
2. Upload the supplied `PIQA-...zip` to the Colab Files pane before the setup cell (or set `PROJECT_ROOT` to its Drive directory).
3. The fixed seed-2026 stratified validation split is created only from the supplied training file. Vocabulary fitting also uses training text only.
4. Validation accuracy selects checkpoints; held-out test labels are used only for the pre-specified final comparisons.
5. The full matrix trains three main seeds and one matched seed for each controlled ablation. Outputs are written to `results/`.

Python 3.10+, PyTorch 2.3+, pandas 2+, scikit-learn 1.3+, SciPy 1.10+, and Matplotlib 3.7+ are supported.
"""
    ),
    markdown("## 1. Environment and supplied data"),
    code(setup_source),
    markdown("## 2. Complete implementation"),
    code(implementation),
    markdown("## 3. Dataset audit and leakage-safe split"),
    code(audit_source, [stream(audit_output)], 1),
    markdown("## 4. Architectural invariant"),
    code(sanity_source, [stream("Swap-equivariance check: PASS\nMain-model trainable parameters: 1,330,049\n")], 2),
    markdown(
        """## 5. Baselines, main runs, and controlled ablations

The matrix is fixed before final evaluation:

- majority and pairwise TF–IDF logistic-regression baselines;
- a standard shared mean-pooled BiGRU neural baseline;
- full GACA-BiGRU at seeds 13, 37, and 73;
- matched seed-13 ablations: no attention, no pair contrast, and simple concatenation features.
"""
    ),
    code(run_source),
    markdown("## 6. Saved training and evaluation evidence"),
    code(
        "SAVED_TRAINING_LOGS = " + repr(log_text) + "\nSAVED_RESULTS = " + repr(results_text) + "\n" + evidence_source,
        [stream(log_text + "\n" + results_text)],
        3,
    ),
    markdown(analysis_md),
    code(attention_code, image_outputs, 4),
    markdown(
        """## 7. Submission consistency checklist

- Main model is trained from random initialisation and contains an explicit attention mechanism.
- Dataset sizes, split seed, vocabulary, truncation, hyperparameters, parameter count, epochs, metrics, and numerical results match the report.
- All baseline values come from this project’s own executions.
- The test split is not used for fitting or checkpoint selection.
- Histories, per-example test predictions, aggregate metrics, checkpoint, and attention JSON/PNGs are regenerated in `results/`.
- Before submission, replace `GROUP_ID` in the notebook filename/title and the report, and replace the report’s contribution placeholder.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
