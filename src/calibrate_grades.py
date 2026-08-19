"""Map labeler grades (0-3 per finding) to probabilities using the gold-58.

Per finding, P(positive | grade) is estimated from gold with Laplace
smoothing. Honest evaluation uses leave-one-out over the gold studies.
Silence (grade 1) gets a finding-specific rate — that is the point: e.g.
unmentioned synovitis is far from certainly negative.

Usage: python calibrate_grades.py train.csv grades.csv out_labels.csv
Prints LOO gold-anchor AUC, writes calibrated soft labels for all graded rows.
"""
import sys

import numpy as np
import pandas as pd

from validate import LABELS, gold_anchor

ALPHA = 1.0  # Laplace smoothing


def fit_map(grades, truth):
    """grade -> smoothed positive rate for one finding."""
    return {g: (truth[grades == g].sum() + ALPHA) / ((grades == g).sum() + 2 * ALPHA)
            for g in (0, 1, 2, 3)}


def main(train_csv, grades_csv, out_csv):
    tr = pd.read_csv(train_csv).set_index("StudyInstanceUID")
    gold = tr.dropna(subset=LABELS)[LABELS]
    gr = pd.read_csv(grades_csv).set_index("StudyInstanceUID")

    common = gold.index.intersection(gr.index)
    print(f"gold overlap: {len(common)}")

    # leave-one-out calibrated predictions on gold
    loo = pd.DataFrame(index=common, columns=LABELS, dtype=float)
    for s in common:
        rest = common.drop(s)
        for c in LABELS:
            m = fit_map(gr.loc[rest, c].values, gold.loc[rest, c].values)
            loo.loc[s, c] = m[int(gr.loc[s, c])]
    r = gold_anchor(gold.loc[common], loo, n_boot=1000)
    print(f"LOO macro={r['macro_auc']:.4f} CI=({r['ci_lo']:.3f},{r['ci_hi']:.3f})")
    print("per-finding:", {k: round(v, 3) for k, v in r["per_finding"].items()})

    # final maps on all gold, applied to every graded study
    out = pd.DataFrame(index=gr.index, columns=LABELS, dtype=float)
    for c in LABELS:
        m = fit_map(gr.loc[common, c].values, gold.loc[common, c].values)
        out[c] = gr[c].astype(int).map(m)
    out.to_csv(out_csv)
    print("wrote", out_csv, len(out), "rows")


if __name__ == "__main__":
    main(*sys.argv[1:4])
