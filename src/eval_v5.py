"""Gate evaluation for labeler v5 (weak-four columns) against the gold-58.

Per weak finding, compares gold AUC (with bootstrap CI) of:
  soft      current v3c-blend soft label column (the incumbent)
  v3sev     our v3 labeler's raw severity
  v5sev     v5 raw severity
  v5cal     v5 grade -> probability, leave-one-out calibrated on gold
  blend     rank-average of v5sev with the incumbent soft column
Decision rule (pre-committed): adopt a v5 column only if v5sev or blend beats
the incumbent by a margin that is not obviously inside the CI noise AND does
not lose on the point estimate; Lateral OA has 11 gold positives, so treat it
as "no-harm" only.

Usage: python eval_v5.py <workdir>   (needs train.csv, soft_labels.csv,
       lab/grades.csv (v3), grades_v5.csv in workdir)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WEAK = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]
ALPHA = 1.0


def auc_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, int), np.asarray(p, float)
    point = roc_auc_score(y, p)
    b = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        if 0 < y[s].mean() < 1:
            b.append(roc_auc_score(y[s], p[s]))
    return point, np.percentile(b, 2.5), np.percentile(b, 97.5)


def loo_calibrated(grades, truth):
    out = np.zeros(len(grades))
    for i in range(len(grades)):
        m = np.ones(len(grades), bool); m[i] = False
        g, t = grades[m], truth[m]
        out[i] = (t[g == grades[i]].sum() + ALPHA) / ((g == grades[i]).sum() + 2 * ALPHA)
    return out


def main(wd):
    wd = Path(wd)
    tr = pd.read_csv(wd / "train.csv").set_index("StudyInstanceUID")
    gold = tr.dropna(subset=WEAK)[WEAK]
    gold = gold[tr.loc[gold.index].notna().all(axis=1)]
    soft = pd.read_csv(wd / "soft_labels.csv").set_index("StudyInstanceUID")
    v3 = pd.read_csv(wd / "lab" / "grades.csv").set_index("StudyInstanceUID")
    v5 = pd.read_csv(wd / "grades_v5.csv").set_index("StudyInstanceUID")
    idx = gold.index.intersection(v5.index)
    weak = [c for c in WEAK if c in v5.columns]
    print(f"gold n={len(idx)}  v5 parse_ok={v5.loc[idx, 'parse_ok'].mean():.3f}")
    rows = []
    for c in weak:
        y = (gold.loc[idx, c] >= 0.5).astype(int).values
        cand = {
            "soft": soft.loc[idx, c].values,
            "v3sev": v3.loc[idx, c + "_sev"].values,
            "v5sev": v5.loc[idx, c + "_sev"].values,
            "v5cal": loo_calibrated(v5.loc[idx, c].values.astype(int), y.astype(float)),
        }
        r_soft = pd.Series(cand["soft"]).rank(pct=True).values
        r_v5 = pd.Series(cand["v5sev"]).rank(pct=True).values
        cand["blend"] = 0.5 * r_soft + 0.5 * r_v5
        for k, p in cand.items():
            pt, lo, hi = auc_ci(y, p)
            rows.append({"finding": c, "npos": int(y.sum()), "set": k,
                         "auc": round(pt, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    df = pd.DataFrame(rows)
    print(df.pivot(index="finding", columns="set", values="auc")[
        ["soft", "v3sev", "v5sev", "v5cal", "blend"]].to_string())
    print()
    print(df[df.set.isin(["soft", "v5sev", "blend"])].to_string(index=False))
    # grade distribution + a few evidence quotes for eyeballing
    for c in weak:
        print(f"\n{c} v5 grade dist on gold:",
              v5.loc[idx, c].value_counts().sort_index().to_dict())
        ev = v5.loc[idx, [c, c + "_sev"] + ([c + "_ev"] if c + "_ev" in v5.columns else [])].copy()
        ev["gold"] = gold.loc[idx, c].values
        print(ev.sort_values(c + "_sev", ascending=False).head(6).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
