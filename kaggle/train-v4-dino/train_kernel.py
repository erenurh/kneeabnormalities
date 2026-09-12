"""v3 = v2 encoder/BiGRU + per-finding query-attention head (single variable vs\ntrain-v2-r3: head only; same 4x24 320px cache, r3 targets, 10 ep, batch 2).

High-res distilled training: 320px cache, all-data student (T4, internet ON
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
import torch.nn.functional as F
from transformers import AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

FOLD = -1  # -1: train on ALL studies (distill targets are OOF-based, leak-free)
EPOCHS = 10
SMOKE = True
UNFREEZE_LAST = 6
DINO_RES = 336
LR_BACKBONE = 1e-5
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
SOFT = (sorted(INPUT.rglob("distill_targets_r3.csv"))
        or sorted(INPUT.rglob("distill_targets_r2.csv"))
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


class DinoEnc(nn.Module):
    """DINOv2-S feature extractor: (N,3,H,W) uint8-normalised in [0,1] ->
    (N, 768) = CLS (+) mean patch token. ImageNet normalisation inside."""

    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained("facebook/dinov2-small")
        for p in self.m.parameters():
            p.requires_grad = False
        for blk in self.m.encoder.layer[-UNFREEZE_LAST:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.m.layernorm.parameters():
            p.requires_grad = True
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.num_features = 2 * self.m.config.hidden_size

    def forward(self, x):
        x = F.interpolate(x, size=(DINO_RES, DINO_RES), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        h = self.m(pixel_values=x).last_hidden_state          # (N,1+P,384)
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)     # (N,768)


class Net(nn.Module):
    """v3 head: per-finding query attention (the public Raptor/DINO recipe's
    'part that matters most'). Encoder + BiGRU-over-triplets as in v2, but the
    shared triplet-attention + shared slot-attention + one MLP are replaced by
    12 learned finding queries attending over ALL (slot x triplet) tokens
    (slot embedding + position embedding + a sex token), then a per-finding
    linear readout. Each finding picks its own planes and slices."""

    def __init__(self, n_slots=N_SLOTS, n_trip=N_SLICES // 3, dim=512, heads=8):
        super().__init__()
        self.enc = DinoEnc()
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
    if SMOKE:
        df = df.sample(300, random_state=0)
    if FOLD < 0:
        trn, val = df, df[df.fold == 0]  # val = fold-0, in-train: logging only, biased
    else:
        trn, val = df[df.fold != FOLD], df[df.fold == FOLD]
    print(f"train {len(trn)} val {len(val)} cache {len(NPZ)} soft={SOFT.name}")

    dev = "cuda"
    net = Net().to(dev)
    bb = [p for n, p in net.named_parameters() if n.startswith("enc.") and p.requires_grad]
    hd = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    print(f"trainable backbone params {sum(p.numel() for p in bb)/1e6:.1f}M, head {sum(p.numel() for p in hd)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": bb, "lr": LR_BACKBONE}, {"params": hd, "lr": LR}],
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * (len(trn) // BATCH + 1))
    scaler = torch.amp.GradScaler()
    dl_t = DataLoader(KneeDS(trn, True), batch_size=BATCH, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)
    dl_v = DataLoader(KneeDS(val, False), batch_size=BATCH, num_workers=4)

    aucs, ep_done = [0.0], 0
    for ep in range(1 if SMOKE else EPOCHS):
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
        torch.save(net.state_dict(), "/kaggle/working/dinov2s_v4_all.pt")
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
