"""Baseline vision training (T4, internet ON for timm weights — allowed for
training kernels; only the submission notebook must be offline).

2.5D: per slot, 12 cached slices -> 4 triplets (3ch) -> shared EfficientNetV2-S
-> mean over triplets -> slot embedding; masked mean over 4 slots + PatientSex
-> 12 logits. Soft-label BCE. Trains FOLD, saves weights + OOF predictions.

Inputs: preprocess cache dataset, folds+labels dataset, competition CSVs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

FOLD = 0
EPOCHS = 6
BATCH = 8
LR = 3e-4
SIZE = 256
N_SLICES = 12
N_SLOTS = 4
SEED = 42
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/train_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/train_series.csv")))[0]
CACHE = sorted(INPUT.rglob("*.npz"))[0].parent
FOLDS = sorted(INPUT.rglob("folds.csv"))[0]
SOFT = sorted(INPUT.rglob("soft_labels.csv"))[0]
torch.manual_seed(SEED)
np.random.seed(SEED)


class KneeDS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        z = np.load(CACHE / f"{r.StudyInstanceUID}.npz")
        vol, mask = z["vol"], z["mask"]  # (4,12,S,S) uint8, (4,)
        if self.train and np.random.rand() < 0.5:
            vol = vol[:, ::-1]  # reverse slice order
        x = torch.from_numpy(np.ascontiguousarray(vol)).float() / 255.0
        y = torch.tensor(r[LABELS].astype(float).values, dtype=torch.float32)
        return x, torch.from_numpy(mask), y, torch.tensor(float(r.sex_m))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnetv2_s", pretrained=True,
                                     num_classes=0, in_chans=3)
        d = self.enc.num_features
        self.head = nn.Sequential(nn.Linear(d + 1, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, len(LABELS)))

    def forward(self, x, mask, sex):
        b, k, n, h, w = x.shape
        trip = x.view(b * k, n // 3, 3, h, w).flatten(0, 1)  # (b*k*4,3,h,w)
        f = self.enc(trip).view(b, k, n // 3, -1).mean(2)    # (b,k,d)
        m = mask.float().unsqueeze(-1)
        emb = (f * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([emb, sex.unsqueeze(-1)], -1))


def main():
    folds = pd.read_csv(FOLDS)
    soft = pd.read_csv(SOFT)
    tr_csv = pd.read_csv(COMP / "train.csv")[["StudyInstanceUID"]]
    meta = pd.read_csv(sorted(INPUT.rglob("series_meta.csv"))[0])
    sex = (meta.groupby("StudyInstanceUID")["PatientSex"].first() == "M") \
        .rename("sex_m").reset_index()
    df = folds.merge(soft, on="StudyInstanceUID").merge(sex, on="StudyInstanceUID")
    df = df[df.StudyInstanceUID.isin(tr_csv.StudyInstanceUID)]
    have = {p.stem for p in CACHE.glob("*.npz")}
    df = df[df.StudyInstanceUID.isin(have)]
    trn, val = df[df.fold != FOLD], df[df.fold == FOLD]
    print(f"train {len(trn)} val {len(val)}")

    dev = "cuda"
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * (len(trn) // BATCH + 1))
    scaler = torch.amp.GradScaler()
    dl_t = DataLoader(KneeDS(trn, True), batch_size=BATCH, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)
    dl_v = DataLoader(KneeDS(val, False), batch_size=BATCH, num_workers=4)

    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for x, m, y, s in dl_t:
            x, m, y, s = x.to(dev), m.to(dev), y.to(dev), s.to(dev)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(
                    net(x, m, s), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        net.eval()
        preds = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, m, y, s in dl_v:
                preds.append(torch.sigmoid(
                    net(x.to(dev), m.to(dev), s.to(dev))).float().cpu())
        p = torch.cat(preds).numpy()
        yv = val[LABELS].values
        from sklearn.metrics import roc_auc_score
        aucs = [roc_auc_score((yv[:, j] >= 0.5).astype(int), p[:, j])
                for j in range(12) if 0 < (yv[:, j] >= 0.5).mean() < 1]
        print(f"ep{ep} loss={tot/len(dl_t):.4f} soft-OOF-AUC={np.mean(aucs):.4f}",
              flush=True)

    torch.save(net.state_dict(), f"/kaggle/working/effv2s_f{FOLD}.pt")
    oof = val[["StudyInstanceUID"]].copy()
    oof[LABELS] = p
    oof.to_csv(f"/kaggle/working/oof_f{FOLD}.csv", index=False)
    json.dump({"fold": FOLD, "soft_oof_auc": float(np.mean(aucs))},
              open("/kaggle/working/metrics.json", "w"))


if __name__ == "__main__":
    main()
