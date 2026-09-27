"""Reproducible PIQA experiments for CITS4012 Assignment 2.

The main model (GACA-BiGRU) is trained from scratch and uses goal-conditioned
additive attention plus a swap-equivariant contrastive candidate scorer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Pip's Windows MKL wheels place runtime DLLs outside the normal DLL search path.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _mkl = Path.home() / "AppData/Roaming/Python/Library/bin"
    if _mkl.exists():
        os.add_dll_directory(str(_mkl))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy import sparse
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_SEED = 2026
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def find_data_dir(root: Path) -> Path:
    candidates = [
        root / "PIQA",
        root / "PIQA-20260927T040220Z-1-001" / "PIQA",
        root,
    ]
    for candidate in candidates:
        if (candidate / "train.jsonl").exists() and (candidate / "train-labels.lst").exists():
            return candidate
    matches = list(root.rglob("train-labels.lst"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError("Could not find PIQA train.jsonl and train-labels.lst beneath " + str(root))


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    records = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = [int(x) for x in (data_dir / f"{split}-labels.lst").read_text(encoding="utf-8").splitlines()]
    if len(records) != len(labels):
        raise ValueError(f"{split}: {len(records)} records but {len(labels)} labels")
    frame = pd.DataFrame(records)
    frame["label"] = labels
    frame["row_id"] = np.arange(len(frame))
    return frame[["row_id", "goal", "sol1", "sol2", "label"]]


def split_data(train_all: pd.DataFrame):
    train_idx, val_idx = train_test_split(
        np.arange(len(train_all)),
        test_size=0.15,
        random_state=PROJECT_SEED,
        stratify=train_all["label"],
    )
    return train_all.iloc[train_idx].reset_index(drop=True), train_all.iloc[val_idx].reset_index(drop=True)


class Vocabulary:
    def __init__(self, texts, max_size=20000, min_freq=2):
        counts = Counter(token for text in texts for token in tokenize(text))
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        words = [word for word, freq in ordered if freq >= min_freq][: max_size - 2]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    def encode(self, text, max_len):
        ids = [self.stoi.get(token, 1) for token in tokenize(text)][:max_len]
        return ids or [1]

    def __len__(self):
        return len(self.itos)


class PIQADataset(Dataset):
    def __init__(self, frame, vocab, max_goal=48, max_solution=64):
        self.frame = frame.reset_index(drop=True)
        self.items = []
        for row in self.frame.itertuples(index=False):
            self.items.append(
                (
                    vocab.encode(row.goal, max_goal),
                    vocab.encode(row.sol1, max_solution),
                    vocab.encode(row.sol2, max_solution),
                    int(row.label),
                    int(row.row_id),
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    goals, sol1s, sol2s, labels, row_ids = zip(*batch)

    def pad(sequences):
        lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
        tensor = torch.zeros(len(sequences), int(lengths.max()), dtype=torch.long)
        for i, seq in enumerate(sequences):
            tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return tensor, lengths

    goal, goal_len = pad(goals)
    sol1, sol1_len = pad(sol1s)
    sol2, sol2_len = pad(sol2s)
    return {
        "goal": goal,
        "goal_len": goal_len,
        "sol1": sol1,
        "sol1_len": sol1_len,
        "sol2": sol2,
        "sol2_len": sol2_len,
        "label": torch.tensor(labels, dtype=torch.long),
        "row_id": torch.tensor(row_ids, dtype=torch.long),
    }


def masked_mean(x, lengths):
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    return (x * mask.unsqueeze(-1)).sum(1) / lengths.clamp_min(1).unsqueeze(-1)


class GACABiGRU(nn.Module):
    """Goal-Aware Contrastive Attention BiGRU.

    mode controls controlled ablations:
      main: full model
      no_attention: mean pool solution states
      no_contrast: score each candidate without rival features
      simple_features: attention retained but only [goal; candidate] features
      mean_bigru: standard shared BiGRU mean-pooling neural baseline
    """

    def __init__(self, vocab_size, emb_dim=96, hidden=64, dropout=0.25, mode="main"):
        super().__init__()
        self.mode = mode
        dim = hidden * 2
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.att_s = nn.Linear(dim, dim, bias=False)
        self.att_g = nn.Linear(dim, dim, bias=True)
        self.att_v = nn.Linear(dim, 1, bias=False)

        if mode == "mean_bigru":
            rep_dim = dim * 4
            scorer_in = rep_dim
        elif mode == "simple_features":
            rep_dim = dim * 2
            scorer_in = rep_dim * 3
        else:
            rep_dim = dim * 4
            scorer_in = rep_dim if mode == "no_contrast" else rep_dim * 3
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        nn.init.uniform_(self.embedding.weight, -0.08, 0.08)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def encode(self, ids, lengths):
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ids.size(1))
        return output

    def attend(self, solution_states, solution_lengths, goal_vector):
        energy = self.att_v(torch.tanh(self.att_s(solution_states) + self.att_g(goal_vector).unsqueeze(1))).squeeze(-1)
        mask = torch.arange(solution_states.size(1), device=solution_states.device)[None, :] < solution_lengths[:, None]
        energy = energy.masked_fill(~mask, -1e9)
        alpha = torch.softmax(energy, dim=1)
        attended = torch.bmm(alpha.unsqueeze(1), solution_states).squeeze(1)
        return attended, alpha

    def candidate_rep(self, goal_vector, solution_states, solution_lengths):
        if self.mode in {"no_attention", "mean_bigru"}:
            solution_vector = masked_mean(solution_states, solution_lengths)
            alpha = None
        else:
            solution_vector, alpha = self.attend(solution_states, solution_lengths, goal_vector)
        if self.mode == "simple_features":
            rep = torch.cat([goal_vector, solution_vector], dim=-1)
        else:
            rep = torch.cat(
                [goal_vector, solution_vector, goal_vector * solution_vector, (goal_vector - solution_vector).abs()], dim=-1
            )
        return rep, alpha

    def forward(self, goal, goal_len, sol1, sol1_len, sol2, sol2_len):
        goal_states = self.encode(goal, goal_len)
        sol1_states = self.encode(sol1, sol1_len)
        sol2_states = self.encode(sol2, sol2_len)
        goal_vector = masked_mean(goal_states, goal_len)
        rep1, alpha1 = self.candidate_rep(goal_vector, sol1_states, sol1_len)
        rep2, alpha2 = self.candidate_rep(goal_vector, sol2_states, sol2_len)

        if self.mode in {"no_contrast", "mean_bigru"}:
            z1, z2 = rep1, rep2
        else:
            product = rep1 * rep2
            z1 = torch.cat([rep1, rep1 - rep2, product], dim=-1)
            z2 = torch.cat([rep2, rep2 - rep1, product], dim=-1)
        logits = torch.cat([self.scorer(self.dropout(z1)), self.scorer(self.dropout(z2))], dim=1)
        return logits, alpha1, alpha2


def loaders(train, val, test, vocab, batch_size=128):
    datasets = [PIQADataset(x, vocab) for x in (train, val, test)]
    return (
        DataLoader(datasets[0], batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0),
        DataLoader(datasets[1], batch_size=batch_size * 2, shuffle=False, collate_fn=collate, num_workers=0),
        DataLoader(datasets[2], batch_size=batch_size * 2, shuffle=False, collate_fn=collate, num_workers=0),
    )


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    labels, predictions, probabilities, row_ids = [], [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _, _ = model(batch["goal"], batch["goal_len"], batch["sol1"], batch["sol1_len"], batch["sol2"], batch["sol2_len"])
        probs = torch.softmax(logits, dim=1)
        total_loss += criterion(logits, batch["label"]).item()
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(probs[:, 1].cpu().tolist())
        row_ids.extend(batch["row_id"].cpu().tolist())
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "labels": labels,
        "predictions": predictions,
        "probability_label1": probabilities,
        "row_ids": row_ids,
    }


def train_neural(train, val, test, vocab, mode, seed, args, output_dir):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders(train, val, test, vocab, args.batch_size)
    model = GACABiGRU(len(vocab), args.embedding_dim, args.hidden_dim, args.dropout, mode=mode).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state, best_val, stale = None, -1.0, 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_items = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(batch["goal"], batch["goal_len"], batch["sol1"], batch["sol1_len"], batch["sol2"], batch["sol2_len"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
            total_items += len(batch["label"])
        val_metrics = predict(model, val_loader, args.device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_items,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(f"{mode:15s} seed={seed:2d} epoch={epoch:02d} train_loss={row['train_loss']:.4f} val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}", flush=True)
        if val_metrics["accuracy"] > best_val + 1e-6:
            best_val = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    val_metrics = predict(model, val_loader, args.device)
    test_metrics = predict(model, test_loader, args.device)
    elapsed = time.time() - start
    stem = f"{mode}_seed{seed}"
    pd.DataFrame(history).to_csv(output_dir / f"{stem}_history.csv", index=False)
    prediction_frame = pd.DataFrame(
        {
            "row_id": test_metrics["row_ids"],
            "gold": test_metrics["labels"],
            "prediction": test_metrics["predictions"],
            "probability_label1": test_metrics["probability_label1"],
        }
    ).sort_values("row_id")
    prediction_frame.to_csv(output_dir / f"{stem}_test_predictions.csv", index=False)
    if mode == "main" and seed == args.seeds[0]:
        torch.save({"model_state": model.state_dict(), "vocab": vocab.itos, "config": vars(args)}, output_dir / "gaca_bigru_best.pt")
    summary = {
        "model": mode,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(pd.DataFrame(history)["val_accuracy"].idxmax() + 1),
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "seconds": elapsed,
    }
    print("FINAL", json.dumps(summary), flush=True)
    return model, summary


def majority_baseline(train, val, test):
    majority = int(train["label"].mode().iloc[0])
    result = {"model": "majority", "seed": PROJECT_SEED, "parameters": 0, "best_epoch": 0}
    for name, frame in [("val", val), ("test", test)]:
        pred = np.full(len(frame), majority)
        result[f"{name}_accuracy"] = accuracy_score(frame["label"], pred)
        result[f"{name}_macro_f1"] = f1_score(frame["label"], pred, average="macro")
    result["seconds"] = 0.0
    return result


def tfidf_baseline(train, val, test, output_dir):
    start = time.time()
    corpus = pd.concat([train["goal"], train["sol1"], train["sol2"]], ignore_index=True)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000, min_df=2, sublinear_tf=True)
    vectorizer.fit(corpus)

    def features(frame):
        g = vectorizer.transform(frame["goal"])
        s1 = vectorizer.transform(frame["sol1"])
        s2 = vectorizer.transform(frame["sol2"])
        # Pairwise difference plus difference in lexical goal overlap.
        return sparse.hstack([s1 - s2, g.multiply(s1) - g.multiply(s2)], format="csr")

    x_train, x_val, x_test = features(train), features(val), features(test)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=PROJECT_SEED, solver="liblinear")
    classifier.fit(x_train, train["label"])
    result = {"model": "tfidf_logreg", "seed": PROJECT_SEED, "parameters": int(x_train.shape[1]), "best_epoch": 0}
    for name, frame, x in [("val", val, x_val), ("test", test, x_test)]:
        pred = classifier.predict(x)
        result[f"{name}_accuracy"] = accuracy_score(frame["label"], pred)
        result[f"{name}_macro_f1"] = f1_score(frame["label"], pred, average="macro")
        if name == "test":
            pd.DataFrame({"row_id": frame["row_id"], "gold": frame["label"], "prediction": pred}).to_csv(
                output_dir / "tfidf_test_predictions.csv", index=False
            )
    result["seconds"] = time.time() - start
    return result


def bootstrap_accuracy_interval(labels, predictions, seed=PROJECT_SEED, n_boot=5000):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        samples[i] = np.mean(labels[idx] == predictions[idx])
    return np.quantile(samples, [0.025, 0.975]).tolist()


@torch.no_grad()
def save_attention_examples(model, frame, vocab, output_dir, device="cpu"):
    loader = DataLoader(PIQADataset(frame, vocab), batch_size=256, shuffle=False, collate_fn=collate)
    metrics = predict(model, loader, device)
    pred_by_id = {rid: pred for rid, pred in zip(metrics["row_ids"], metrics["predictions"])}
    confidence_by_id = {
        rid: (p if pred == 1 else 1 - p)
        for rid, pred, p in zip(metrics["row_ids"], metrics["predictions"], metrics["probability_label1"])
    }
    correct = [int(rid) for rid, gold in zip(frame["row_id"], frame["label"]) if pred_by_id[int(rid)] == int(gold)]
    wrong = [int(rid) for rid, gold in zip(frame["row_id"], frame["label"]) if pred_by_id[int(rid)] != int(gold)]
    chosen = [max(correct, key=confidence_by_id.get), max(wrong, key=confidence_by_id.get)]
    examples = []
    model.eval()
    for kind, rid in zip(["correct", "error"], chosen):
        row = frame.loc[frame["row_id"] == rid].iloc[0]
        one = collate([PIQADataset(pd.DataFrame([row]), vocab)[0]])
        one = {k: v.to(device) for k, v in one.items()}
        logits, a1, a2 = model(one["goal"], one["goal_len"], one["sol1"], one["sol1_len"], one["sol2"], one["sol2_len"])
        pred = int(logits.argmax(1).item())
        probs = torch.softmax(logits, 1).cpu().numpy()[0].tolist()
        tokens1, tokens2 = tokenize(row["sol1"])[:64], tokenize(row["sol2"])[:64]
        weights1 = a1[0, : len(tokens1)].cpu().numpy()
        weights2 = a2[0, : len(tokens2)].cpu().numpy()
        example = {
            "kind": kind,
            "row_id": rid,
            "goal": row["goal"],
            "sol1": row["sol1"],
            "sol2": row["sol2"],
            "gold": int(row["label"]),
            "prediction": pred,
            "probabilities": probs,
            "tokens1": tokens1,
            "attention1": weights1.tolist(),
            "tokens2": tokens2,
            "attention2": weights2.tolist(),
        }
        examples.append(example)

        fig, axes = plt.subplots(2, 1, figsize=(10, 3.5), constrained_layout=True)
        for ax, tokens, weights, title in [
            (axes[0], tokens1, weights1, "Solution 1"),
            (axes[1], tokens2, weights2, "Solution 2"),
        ]:
            shown = min(len(tokens), 28)
            ax.imshow(weights[:shown][None, :], aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(weights.max(), 0.15))
            ax.set_xticks(np.arange(shown), tokens[:shown], rotation=45, ha="right", fontsize=8)
            ax.set_yticks([])
            ax.set_ylabel(title, rotation=0, labelpad=40, va="center")
        fig.suptitle(f"{kind.title()} example: gold={int(row['label'])}, prediction={pred}")
        fig.savefig(output_dir / f"attention_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    (output_dir / "attention_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")


def dataset_statistics(train_all, train, val, test, vocab):
    def lens(series):
        values = series.map(lambda x: len(tokenize(x)))
        return {"mean": float(values.mean()), "median": float(values.median()), "p95": float(values.quantile(0.95)), "max": int(values.max())}
    return {
        "sizes": {"provided_train": len(train_all), "train": len(train), "validation": len(val), "test": len(test)},
        "label1_fraction": {"train": float(train["label"].mean()), "validation": float(val["label"].mean()), "test": float(test["label"].mean())},
        "token_lengths": {"goal": lens(train_all["goal"]), "solution1": lens(train_all["sol1"]), "solution2": lens(train_all["sol2"])},
        "vocabulary_size": len(vocab),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 73])
    parser.add_argument("--quick", action="store_true", help="One epoch/seed smoke test; not for reported results")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.patience, args.seeds = 1, 1, [13]
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir(args.root)
    train_all, test = load_split(data_dir, "train"), load_split(data_dir, "test")
    train, val = split_data(train_all)
    vocab = Vocabulary(pd.concat([train["goal"], train["sol1"], train["sol2"]], ignore_index=True))
    stats = dataset_statistics(train_all, train, val, test, vocab)
    (args.output_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(json.dumps({**vars(args), "root": str(args.root), "output_dir": str(args.output_dir)}, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)

    summaries = [majority_baseline(train, val, test), tfidf_baseline(train, val, test, args.output_dir)]
    main_model = None
    for seed in args.seeds:
        model, summary = train_neural(train, val, test, vocab, "main", seed, args, args.output_dir)
        summaries.append(summary)
        if seed == args.seeds[0]:
            main_model = model
    for mode in ["mean_bigru", "no_attention", "no_contrast", "simple_features"]:
        _, summary = train_neural(train, val, test, vocab, mode, args.seeds[0], args, args.output_dir)
        summaries.append(summary)

    results = pd.DataFrame(summaries)
    results.to_csv(args.output_dir / "all_results.csv", index=False)
    main_rows = results[results["model"] == "main"]
    aggregate = {
        "main_test_accuracy_mean": float(main_rows["test_accuracy"].mean()),
        "main_test_accuracy_std": float(main_rows["test_accuracy"].std(ddof=1)) if len(main_rows) > 1 else 0.0,
        "main_test_macro_f1_mean": float(main_rows["test_macro_f1"].mean()),
        "main_test_macro_f1_std": float(main_rows["test_macro_f1"].std(ddof=1)) if len(main_rows) > 1 else 0.0,
    }
    pred_file = args.output_dir / f"main_seed{args.seeds[0]}_test_predictions.csv"
    pred_frame = pd.read_csv(pred_file)
    aggregate["main_seed13_accuracy_95ci"] = bootstrap_accuracy_interval(pred_frame["gold"], pred_frame["prediction"])
    (args.output_dir / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    save_attention_examples(main_model, test, vocab, args.output_dir, args.device)
    print("\nRESULTS\n", results.to_string(index=False), flush=True)
    print("\nAGGREGATE\n", json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
