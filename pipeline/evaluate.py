"""
STEP 3 — Evaluate Predictions and Save Report
==============================================
Input:  facebook-data/dev.jsonl                              (ground-truth labels)
        output/preds_qwen3vl_8b_4bit_siglip_rag.jsonl       (step 2 predictions)
         — OR any other preds_*.jsonl file in output/

Output: output/eval_report.json   full metrics for every prediction file found
        stdout                    human-readable table

Metrics computed per file:
  • Accuracy
  • Precision  (positive class = hateful, label=1)
  • Recall
  • F1-score
  • AUROC       (based on raw label values; binary so 0/1 scores are used)

The script auto-discovers all preds_*.jsonl files in output/ so adding new
model results requires no code changes — just re-run step3.
"""

import json
import glob
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[1]
GT_FILE    = BASE / "facebook-data" / "dev.jsonl"   # has label field
OUTPUT_DIR = BASE / "output"
REPORT     = OUTPUT_DIR / "eval_report.json"


# ── Load ground truth ─────────────────────────────────────────────────────────

def load_ground_truth(path: Path) -> dict:
    """Returns {str(id): int(label)} for every sample that has a label."""
    gt = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "label" in item:
                gt[str(item["id"])] = int(item["label"])
    return gt


# ── Evaluate one prediction file ─────────────────────────────────────────────

def evaluate_file(pred_path: Path, gt: dict) -> dict | None:
    """
    Load predictions, align with ground truth by id, compute metrics.
    Returns a metrics dict or None if there are no matching samples.
    """
    y_true, y_pred = [], []

    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            id_str = str(item.get("id", ""))
            if id_str in gt:
                y_true.append(gt[id_str])
                y_pred.append(int(item.get("label", 0)))

    if not y_true:
        return None

    # AUROC requires at least one sample of each class
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = None

    return {
        "n_samples":  len(y_true),
        "accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auroc":      round(auroc, 4) if auroc is not None else "n/a",
    }


# ── Pretty table ──────────────────────────────────────────────────────────────

def print_table(results: dict):
    if not results:
        print("No results to display.")
        return

    cols   = ["Model", "N", "Accuracy", "Precision", "Recall", "F1", "AUROC"]
    widths = [max(len(c), 40) for c in cols]
    widths[0] = 45

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep    = "  ".join("-" * w for w in widths)
    print("\n" + header)
    print(sep)

    for model_name, m in sorted(results.items()):
        row = [
            model_name,
            str(m["n_samples"]),
            str(m["accuracy"]),
            str(m["precision"]),
            str(m["recall"]),
            str(m["f1"]),
            str(m["auroc"]),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 3 — Evaluation")
    print("=" * 60)

    gt = load_ground_truth(GT_FILE)
    print(f"Ground truth: {len(gt)} labelled samples from {GT_FILE.name}")

    pred_files = sorted(OUTPUT_DIR.glob("preds_*.jsonl"))
    if not pred_files:
        print(f"\nNo preds_*.jsonl files found in {OUTPUT_DIR}. Run step2 first.")
        return

    print(f"Found {len(pred_files)} prediction file(s):\n")
    for p in pred_files:
        print(f"  {p.name}")

    all_results = {}
    for pred_path in pred_files:
        model_name = pred_path.stem.removeprefix("preds_")
        metrics = evaluate_file(pred_path, gt)
        if metrics is None:
            print(f"\n  [skip] {pred_path.name} — no samples matched ground truth IDs")
            continue
        all_results[model_name] = metrics
        print(f"\n{model_name}")
        for k, v in metrics.items():
            print(f"  {k:<12} {v}")

    # Save JSON report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report saved → {REPORT}")

    # Print summary table
    print_table(all_results)
    print()


if __name__ == "__main__":
    main()