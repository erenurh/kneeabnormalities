"""High-res distilled training: 320px cache, all-data student (T4, internet ON
for timm weights; only the submission notebook must be offline).

Same recipe as the 0.910-LB 10-epoch all-data distilled model, with SIZE
256->320 (single variable). Cache comes from the two CPU half-kernels
(preprocess-320a/b), so npz files live in two input dirs -> uid->path map.

Slices 15->24 (single variable vs the 0.915 r2 student). Batch 2 at 320px (14.5GB T4; 4*4slots*5triplets*320^2 ~ memory of the
proven 256px config), LR scaled with batch. Time guard: T4 sessions cap at
12h and a killed kernel loses outputs, so past 10.5h we stop after the
current epoch and save whatever is trained.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

FOLD = 3
EPOCHS = 10
BATCH = 2
LR = 1e-4
SIZE = 320
N_SLICES = 24  # cache has 24 -> 8 triplets per slot
N_SLOTS = 4
SEED = 42
TIME_BUDGET_S = 10.5 * 3600
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/train_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/train_series.csv")))[0]
NPZ = {p.stem: p for p in INPUT.rglob("*.npz")}  # spans both half caches
FOLDS = sorted(INPUT.rglob("folds.csv"))[0]
SOFT = (sorted(INPUT.rglob("distill_targets_r2.csv"))
        or sorted(INPUT.rglob("distill_targets.csv"))
        or sorted(INPUT.rglob("report_labels_v4hybrid.csv")))[0]
torch.manual_seed(SEED)
np.random.seed(SEED)
START = time.time()


class KneeDS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        z = np.load(NPZ[r.StudyInstanceUID])
        vol, mask = z["vol"][:, :N_SLICES], z["mask"]  # cache (4,24,S,S)
        if self.train and np.random.rand() < 0.5:
            vol = vol[:, ::-1]  # reverse slice order (exp-6 showed per-plane
            # label-swap variants hurt: swapping only sagittal labels
            # contradicts the unmirrored coronal/axial slots)
        x = torch.from_numpy(np.ascontiguousarray(vol)).float() / 255.0
        y = torch.tensor(r[LABELS].astype(float).values, dtype=torch.float32)
        return x, torch.from_numpy(mask), y, torch.tensor(float(r.sex_m))


class Net(nn.Module):
    """RSNA-winner aggregation: BiGRU over slice triplets + attention-MIL
    pooling per series, masked attention over the 4 series slots (replaces
    the plain means of the 0.915 baseline; encoder unchanged)."""

    def __init__(self):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnetv2_s", pretrained=True,
                                     num_classes=0, in_chans=3)
        d = self.enc.num_features
        self.gru = nn.GRU(d, 256, batch_first=True, bidirectional=True)
        self.trip_att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(),
                                      nn.Linear(128, 1))
        self.slot_att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(),
                                      nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Linear(512 + 1, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, len(LABELS)))

    def forward(self, x, mask, sex):
        b, k, n, h, w = x.shape
        t = n // 3
        trip = x.view(b * k, t, 3, h, w).flatten(0, 1)
        f = self.enc(trip).view(b * k, t, -1)               # (b*k,t,d)
        hseq, _ = self.gru(f)                               # (b*k,t,512)
        wt = torch.softmax(self.trip_att(hseq), dim=1)
        slot = (wt * hseq).sum(1).view(b, k, -1)            # (b,k,512)
        ws = self.slot_att(slot).squeeze(-1)                # (b,k)
        ws = ws.masked_fill(~mask, float("-inf")).softmax(-1).unsqueeze(-1)
        emb = (ws * slot).sum(1)
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
    df = df[df.StudyInstanceUID.isin(NPZ)]
    if FOLD < 0:
        trn, val = df, df[df.fold == 0]  # val = fold-0, in-train: logging only, biased
    else:
        trn, val = df[df.fold != FOLD], df[df.fold == FOLD]
    print(f"train {len(trn)} val {len(val)} cache {len(NPZ)} soft={SOFT.name}")

    dev = "cuda"
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * (len(trn) // BATCH + 1))
    scaler = torch.amp.GradScaler()
    dl_t = DataLoader(KneeDS(trn, True), batch_size=BATCH, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)
    dl_v = DataLoader(KneeDS(val, False), batch_size=BATCH, num_workers=4)

    aucs, ep_done = [0.0], 0
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
        ep_done = ep + 1
        elapsed = time.time() - START
        print(f"ep{ep} loss={tot/len(dl_t):.4f} soft-OOF-AUC={np.mean(aucs):.4f}"
              f" elapsed={elapsed/3600:.2f}h", flush=True)
        torch.save(net.state_dict(), "/kaggle/working/effv2s_v2_f3.pt")
        if elapsed > TIME_BUDGET_S:
            print("time budget hit, stopping early", flush=True)
            break

    oof = val[["StudyInstanceUID"]].copy()
    oof[LABELS] = p
    oof.to_csv("/kaggle/working/val_preds.csv", index=False)
    json.dump({"fold": FOLD, "size": SIZE, "epochs_done": ep_done,
               "soft_oof_auc": float(np.mean(aucs))},
              open("/kaggle/working/metrics.json", "w"))


if __name__ == "__main__":
    main()
