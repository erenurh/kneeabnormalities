"""Evaluate candidate report-label sets against the gold-58 anchor.

Usage: python eval_labels.py train.csv labels1.csv [labels2.csv ...]
Each label CSV needs StudyInstanceUID + the 12 label columns.
Prints macro-AUC with bootstrap CI and per-finding AUCs per set, plus the
pairwise rank-average blends. Label sets that contain the gold labels
themselves (macro ≈ 1.0) are leaks, not labelers — discard them.
"""
import sys

import pandas as pd

from validate import LABELS, gold_anchor


def main(train_csv, *label_csvs):
    tr = pd.read_csv(train_csv).set_index("StudyInstanceUID")
    gold = tr.dropna(subset=LABELS)[LABELS]
    preds = {}
    for f in label_csvs:
        df = pd.read_csv(f).set_index("StudyInstanceUID")[LABELS].astype(float)
        preds[f] = df
        r = gold_anchor(gold, df.reindex(gold.index).fillna(0.5), n_boot=1000)
        print(f"{f}\n  macro={r['macro_auc']:.4f} CI=({r['ci_lo']:.3f},{r['ci_hi']:.3f})")
        print("  per-finding:", {k: round(v, 3) for k, v in r["per_finding"].items()})
    if len(preds) > 1:
        ranked = [df.reindex(tr.index).rank(pct=True) for df in preds.values()]
        b = sum(ranked) / len(ranked)
        r = gold_anchor(gold, b.loc[gold.index].fillna(0.5), n_boot=1000)
        print(f"rank-blend-all macro={r['macro_auc']:.4f} CI=({r['ci_lo']:.3f},{r['ci_hi']:.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
