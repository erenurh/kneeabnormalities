"""Two-tier validation harness.

Tier 1 (volume): OOF macro-AUC of predictions vs soft report labels.
Tier 2 (anchor): macro-AUC vs the 58 gold studies, with bootstrap CIs.

All functions take aligned DataFrames indexed by StudyInstanceUID with the
12 label columns. The gold anchor is the only unbiased signal; its CI is
wide (n=58) — decisions require CI-separated wins, not point estimates.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def per_finding_auc(y_true, y_pred, threshold=0.5):
    """AUC per finding; soft truths are binarized at `threshold`. NaN if degenerate."""
    out = {}
    for c in LABELS:
        t, p = y_true[c].values, y_pred[c].values
        keep = ~np.isnan(t)
        tb = (t[keep] >= threshold).astype(int)
        out[c] = roc_auc_score(tb, p[keep]) if 0 < tb.mean() < 1 else np.nan
    return pd.Series(out)


def macro_auc(y_true, y_pred, threshold=0.5):
    return float(per_finding_auc(y_true, y_pred, threshold).mean())


def gold_anchor(y_gold, y_pred, n_boot=2000, seed=0):
    """Macro-AUC on the gold set with a study-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    idx = y_gold.index.to_numpy()
    point = macro_auc(y_gold, y_pred.loc[idx])
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_gold.loc[s], y_pred.loc[s]
        if all((yt[c] >= 0.5).astype(int).nunique() > 1 for c in LABELS):
            boots.append(macro_auc(yt, yp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"macro_auc": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot_valid": len(boots),
            "per_finding": per_finding_auc(y_gold, y_pred.loc[idx]).round(4).to_dict()}


def report(oof_true, oof_pred, gold_true, gold_pred):
    """One dashboard dict per experiment; log it with the run config."""
    return {
        "tier1_soft_oof_macro": macro_auc(oof_true, oof_pred),
        "tier1_per_finding": per_finding_auc(oof_true, oof_pred).round(4).to_dict(),
        "tier2_gold": gold_anchor(gold_true, gold_pred),
    }
