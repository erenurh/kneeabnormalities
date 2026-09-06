"""Build distill-round-3 targets from the v2 fold models' OOF predictions.

r3 = 0.5 * merged v2 5-fold OOF + 0.5 * v3c-blend soft labels (same recipe as
rounds 1-2; teacher upgraded from the plain-320 folds to the v2 attention
folds). Gate: gold-58 ranking AUC must be >= r2's 0.9237.
Measured 09-06: r3 0.9266 vs r2 0.9237 (PASS).

Inputs (local work dir, small CSVs only, never committed):
  f0..f4/val_preds.csv  v2 fold OOF preds; soft_labels.csv; distill_targets_r2.csv; train.csv
Usage: python build_distill_targets_r3.py <workdir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def gold_auc(t, gold):
    m = gold.merge(t, on="StudyInstanceUID", suffixes=("_g", "_t"))
    return float(np.mean([roc_auc_score(m[f"{c}_g"].astype(int), m[f"{c}_t"])
                          for c in LABELS if 0 < m[f"{c}_g"].mean() < 1]))


def main(wd):
    wd = Path(wd)
    oof = pd.concat([pd.read_csv(wd / f"f{k}" / "val_preds.csv") for k in range(5)])
    assert oof.StudyInstanceUID.is_unique
    soft = pd.read_csv(wd / "soft_labels.csv")
    r2 = pd.read_csv(wd / "distill_targets_r2.csv")
    train = pd.read_csv(wd / "train.csv")
    gold = train[train[LABELS].notna().all(axis=1)][["StudyInstanceUID"] + LABELS]
    m = soft.merge(oof, on="StudyInstanceUID", how="left", suffixes=("_soft", "_oof"))
    cov = m[f"{LABELS[0]}_oof"].notna()
    print(f"oof {len(oof)} soft {len(soft)} coverage {cov.sum()}/{len(m)}")
    r3 = m[["StudyInstanceUID"]].copy()
    for c in LABELS:
        r3[c] = np.where(cov, 0.5 * m[f"{c}_oof"] + 0.5 * m[f"{c}_soft"], m[f"{c}_soft"])
    a2, a3 = gold_auc(r2, gold), gold_auc(r3, gold)
    print(f"gold-{len(gold)} ranking AUC  r2={a2:.4f}  r3={a3:.4f}  gate={'PASS' if a3 >= a2 else 'FAIL'}")
    r3.to_csv(wd / "distill_targets_r3.csv", index=False)
    print("wrote", wd / "distill_targets_r3.csv")


if __name__ == "__main__":
    main(sys.argv[1])
