"""Build distill-round-2 targets from the 320px fold models' OOF predictions.

r2 targets = 0.5 * merged-fold-OOF + 0.5 * v3c-blend soft labels (same recipe
as round 1, teacher upgraded from 256px folds to 320px distill-trained folds).
Gate: gold-58 ranking AUC must be >= round-1 targets' 0.9166 before the r2
student is trained.

Inputs live in a local work dir (small CSVs only, never committed):
  f0..f4/val_preds.csv  fold OOF preds (StudyInstanceUID + 12 label cols)
  soft_labels.csv       v3c-blend soft labels
  distill_targets.csv   round-1 targets (for the side-by-side gold eval)
  train.csv             competition labels; the 58 fully-labeled rows = gold-58

Usage: python build_distill_targets_r2.py <workdir> [n_folds]
Writes <workdir>/distill_targets_r2.csv and prints the gate numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def gold_auc(targets, gold):
    m = gold.merge(targets, on="StudyInstanceUID", suffixes=("_g", "_t"))
    aucs = []
    for c in LABELS:
        y = m[f"{c}_g"].values.astype(int)
        if 0 < y.mean() < 1:
            aucs.append(roc_auc_score(y, m[f"{c}_t"].values))
    return float(np.mean(aucs)), len(m)


def main():
    wd = Path(sys.argv[1])
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    oof = pd.concat([pd.read_csv(wd / f"f{k}" / "val_preds.csv")
                     for k in range(n_folds)], ignore_index=True)
    assert oof.StudyInstanceUID.is_unique, "fold OOFs overlap"
    soft = pd.read_csv(wd / "soft_labels.csv")
    r1 = pd.read_csv(wd / "distill_targets.csv")
    train = pd.read_csv(wd / "train.csv")
    gold = train[train[LABELS].notna().all(axis=1)][["StudyInstanceUID"] + LABELS]
    print(f"oof {len(oof)} soft {len(soft)} gold {len(gold)}")

    m = soft.merge(oof, on="StudyInstanceUID", how="left",
                   suffixes=("_soft", "_oof"))
    covered = m[f"{LABELS[0]}_oof"].notna()
    print(f"OOF coverage {covered.sum()}/{len(m)}")
    r2 = m[["StudyInstanceUID"]].copy()
    for c in LABELS:
        # studies without an OOF row (shouldn't happen at 5 folds) fall back to soft
        r2[c] = np.where(covered, 0.5 * m[f"{c}_oof"] + 0.5 * m[f"{c}_soft"],
                         m[f"{c}_soft"])

    a1, n1 = gold_auc(r1, gold)
    a2, n2 = gold_auc(r2, gold)
    print(f"gold-{n1} ranking AUC  r1={a1:.4f}  r2={a2:.4f}  "
          f"gate={'PASS' if a2 >= a1 else 'FAIL'}")
    r2.to_csv(wd / "distill_targets_r2.csv", index=False)
    print("wrote", wd / "distill_targets_r2.csv")


if __name__ == "__main__":
    main()
