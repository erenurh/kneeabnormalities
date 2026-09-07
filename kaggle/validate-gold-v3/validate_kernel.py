"""Two-tier validation, tier 2: gold-58 anchor for the v3 checkpoint
(per-finding query attention, train-v3); v2 result to compare: 0.9218.

Inference-only against studies never trained on for their labels (the
model trains on soft/distill targets for ALL studies including gold rows,
but gold rows' hard ground truth in train.csv was never used as a loss
target anywhere in this line -- only as the eval signal here and in
src/folds.py's fold assignment). Offline, no timm download: the encoder
is built with pretrained=False since the full trained state_dict is
loaded right after, so internet stays off and this stays fast/cheap on
the near-empty GPU quota.

Mirrors src/validate.py's gold_anchor() so the numbers are directly
comparable to that harness; keep the two in sync if either changes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SIZE = 320
N_SLICES = 24
N_SLOTS = 4
BATCH = 4
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/train_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/train_series.csv")))[0]
NPZ = {p.stem: p for p in INPUT.rglob("*.npz")}
FOLDS = sorted(INPUT.rglob("folds.csv"))[0]
CKPT = sorted(INPUT.rglob("effv2s_v3_all.pt"))[0]


class GoldDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        z = np.load(NPZ[r.StudyInstanceUID])
        vol, mask = z["vol"][:, :N_SLICES], z["mask"]
        x = torch.from_numpy(np.ascontiguousarray(vol)).float() / 255.0
        y = torch.tensor(r[LABELS].astype(float).values, dtype=torch.float32)
        return x, torch.from_numpy(mask), y, torch.tensor(float(r.sex_m))


class Net(nn.Module):
    """v3 head: per-finding query attention (the public Raptor/DINO recipe's
    'part that matters most'). Encoder + BiGRU-over-triplets as in v2, but the
    shared triplet-attention + shared slot-attention + one MLP are replaced by
    12 learned finding queries attending over ALL (slot x triplet) tokens
    (slot embedding + position embedding + a sex token), then a per-finding
    linear readout. Each finding picks its own planes and slices."""

    def __init__(self, n_slots=N_SLOTS, n_trip=N_SLICES // 3, dim=512, heads=8):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnetv2_s", pretrained=False,
                                     num_classes=0, in_chans=3)
        d = self.enc.num_features
        self.gru = nn.GRU(d, dim // 2, batch_first=True, bidirectional=True)
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, dim))
        self.pos_emb = nn.Parameter(torch.zeros(n_trip, dim))
        self.sex_emb = nn.Parameter(torch.zeros(2, dim))
        self.queries = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.att = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.sex_emb, std=0.02)

    def forward(self, x, mask, sex):
        b, k, n, h, w = x.shape
        t = n // 3
        trip = x.view(b * k, t, 3, h, w).flatten(0, 1)
        f = self.enc(trip).view(b * k, t, -1)               # (b*k,t,d)
        hseq, _ = self.gru(f)                               # (b*k,t,dim)
        tok = hseq.view(b, k, t, -1) + self.slot_emb[None, :, None] \
            + self.pos_emb[None, None]
        tok = tok.reshape(b, k * t, -1)
        tokmask = mask.unsqueeze(-1).expand(b, k, t).reshape(b, k * t)
        sex_tok = self.sex_emb[sex.long()].unsqueeze(1)     # (b,1,dim)
        tok = torch.cat([tok, sex_tok], 1)
        tokmask = torch.cat([tokmask, torch.ones(b, 1, dtype=torch.bool,
                                                 device=tok.device)], 1)
        tok = self.norm(tok)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)      # (b,12,dim)
        o, _ = self.att(q, tok, tok, key_padding_mask=~tokmask)
        o = self.drop(self.out_norm(o + q))                  # (b,12,dim)
        return (o * self.cls_w).sum(-1) + self.cls_b          # (b,12)


def gold_anchor(y_true, y_pred, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = macro_auc(y_true, y_pred)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if all(0 < (yt[:, j] >= 0.5).mean() < 1 for j in range(len(LABELS))):
            boots.append(macro_auc(yt, yp))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"macro_auc": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot_valid": len(boots),
            "per_finding": {k: (None if np.isnan(v) else round(v, 4))
                            for k, v in per_finding_auc(y_true, y_pred).items()}}


def main():
    folds = pd.read_csv(FOLDS)
    tr = pd.read_csv(COMP / "train.csv")
    meta = pd.read_csv(sorted(INPUT.rglob("series_meta.csv"))[0])
    sex = (meta.groupby("StudyInstanceUID")["PatientSex"].first() == "M") \
        .rename("sex_m").reset_index()

    is_gold = tr[LABELS].notna().all(axis=1)
    gold = tr[is_gold].merge(sex, on="StudyInstanceUID")
    gold = gold[gold.StudyInstanceUID.isin(NPZ)]
    print(f"gold studies with cache: {len(gold)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()

    dl = DataLoader(GoldDS(gold), batch_size=BATCH, num_workers=2)
    preds = []
    with torch.no_grad(), torch.amp.autocast(dev if dev == "cuda" else "cpu"):
        for x, m, y, s in dl:
            preds.append(torch.sigmoid(
                net(x.to(dev), m.to(dev), s.to(dev))).float().cpu())
    p = torch.cat(preds).numpy()
    yt = gold[LABELS].values.astype(float)

    result = gold_anchor(yt, p)
    print(json.dumps(result, indent=2), flush=True)
    json.dump({"checkpoint": CKPT.name, "n_gold": len(gold), **result},
              open("/kaggle/working/gold_result.json", "w"))

    out = gold[["StudyInstanceUID"]].copy()
    out[LABELS] = p
    out.to_csv("/kaggle/working/gold_preds.csv", index=False)


if __name__ == "__main__":
    main()
