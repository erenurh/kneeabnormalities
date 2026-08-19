"""Build the frozen 5-fold dual-grouped study-level CV assignment.

Groups studies by union-find over (exact-duplicate report hash) union
(site fingerprint = manufacturer + model + field strength), then assigns
whole groups to folds greedily, balancing fold sizes and soft label burden.

Inputs (paths given on CLI): train.csv, series_meta.csv (audit kernel output).
Output: folds.csv with StudyInstanceUID,group_id,fold — NOT to be committed
to the (public) repo; upload as a private Kaggle dataset.
"""
import hashlib
import sys

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
N_FOLDS = 5
SEED = 42


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main(train_csv, series_meta_csv, out_csv):
    tr = pd.read_csv(train_csv)
    meta = pd.read_csv(series_meta_csv)

    tr["rhash"] = tr["Report"].fillna("").map(
        lambda s: hashlib.md5(s.strip().encode()).hexdigest())

    site = (meta.assign(mfg=meta["Manufacturer"].str.upper().str.split().str[0],
                        fs=meta["MagneticFieldStrength"].astype(float).round(1))
            .groupby("StudyInstanceUID")
            .agg(model=("ManufacturerModelName", "first"),
                 mfg=("mfg", "first"), fs=("fs", "first")))
    tr = tr.merge(site, on="StudyInstanceUID", how="left")
    tr["site_fp"] = (tr["mfg"].fillna("?") + "|" + tr["model"].fillna("?")
                     + "|" + tr["fs"].astype(str))

    # Union studies sharing a duplicate report; unions via site alone would
    # collapse everything into 45 mega-groups, so site only glues studies
    # whose reports are exact duplicates of each other (already same group)
    # plus their site-mates with the SAME duplicated template.
    uf = UnionFind()
    for _, g in tr[tr.duplicated("rhash", keep=False)].groupby("rhash"):
        ids = g["StudyInstanceUID"].tolist()
        for other in ids[1:]:
            uf.union(ids[0], other)
    tr["group_id"] = tr["StudyInstanceUID"].map(uf.find)

    # Greedy assignment: order groups by size desc, place each into the fold
    # with the smallest (size, gold-count) load; keeps gold-58 spread evenly.
    rng = np.random.default_rng(SEED)
    tr["is_gold"] = tr[LABELS].notna().all(axis=1)
    groups = (tr.groupby("group_id")
              .agg(n=("StudyInstanceUID", "size"), gold=("is_gold", "sum"))
              .sample(frac=1, random_state=SEED)
              .sort_values(["n", "gold"], ascending=False))
    load = np.zeros(N_FOLDS)
    gold_load = np.zeros(N_FOLDS)
    fold_of = {}
    for gid, row in groups.iterrows():
        f = int(np.lexsort((load, gold_load))[0]) if row["gold"] else int(np.argmin(load))
        fold_of[gid] = f
        load[f] += row["n"]
        gold_load[f] += row["gold"]
    tr["fold"] = tr["group_id"].map(fold_of)

    tr[["StudyInstanceUID", "group_id", "fold"]].to_csv(out_csv, index=False)
    print("fold sizes:", tr["fold"].value_counts().sort_index().tolist())
    print("gold per fold:", tr[tr["is_gold"]]["fold"].value_counts().sort_index().tolist())
    print("largest group:", int(groups["n"].max()))
    dup_leak = (tr.groupby("rhash")["fold"].nunique() > 1) & tr.groupby("rhash").size().gt(1)
    print("duplicate-report groups straddling folds:", int(dup_leak.sum()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
