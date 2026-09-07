"""Efficiency-track RUNTIME HARNESS (offline, T4). Not a submission.

Runs the exact submit-v2 pipeline (same preprocessing + v2 checkpoint) over
N_STUDIES training studies and times it two ways: (A) sequential, as the
submitted kernel does it; (B) DICOM preprocessing in a 4-process pool feeding
the GPU. Extrapolates both to the ~1,300-study hidden test so the efficiency
entry's runtime term (RuntimeSeconds/32400; 0.01 AUC ~ 720 s) is measured,
not guessed. Original docstring of the mirrored submit kernel follows.

Submission kernel (offline, T4). Produces /kaggle/working/submission.csv.

Mirrors the training preprocessing exactly (position-sorted slices, 140mm
center crop, per-slice percentile norm, 4 slots with header-derived fat-sat)
and runs the v2 checkpoint (24 slices, BiGRU+attention aggregation, train-v2).
Any study that fails preprocessing gets the training-prevalence prior so the
submission always has every required row.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from PIL import Image

N_SLICES = 24
SIZE = 320
CROP_MM = 140.0
SLOTS = [("Sagittal", True), ("Sagittal", False), ("Coronal", True), ("Axial", True)]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PRIOR = [0.30, 0.12, 0.40, 0.30, 0.25, 0.18, 0.30, 0.45, 0.35, 0.18, 0.25, 0.20]
FS_PAT = re.compile(r"(?i)\bfs\b|fat.?sat|spair|spir|stir|tirm|_fs|fs_|fatsat")

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/train_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/train_series.csv")))[0]
SPLIT = "train_series"  # harness reads training studies
N_STUDIES = 60
HIDDEN_N = 1300  # approx hidden test size for extrapolation
CKPT = sorted(INPUT.rglob("effv2s_v2_all.pt"))[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def plane_of(iop):
    n = np.abs(np.cross(np.array(iop[:3], float), np.array(iop[3:], float)))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]


def series_info(sdir):
    f = next(sdir.glob("*.dcm"), None)
    if f is None:
        return None
    ds = pydicom.dcmread(f, stop_before_pixels=True)
    iop = getattr(ds, "ImageOrientationPatient", None)
    return {
        "plane": plane_of(iop) if iop is not None else None,
        "fatsat": bool(FS_PAT.search(
            f"{getattr(ds, 'ScanOptions', '')} {getattr(ds, 'SeriesDescription', '')}")),
        "n": len(list(sdir.glob("*.dcm"))),
        "sex": getattr(ds, "PatientSex", ""),
        "dir": sdir,
    }


def load_series(sdir, n):
    heads = []
    for f in sdir.glob("*.dcm"):
        ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=[
            "ImagePositionPatient", "ImageOrientationPatient"])
        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if iop is None or ipp is None:
            return None
        nvec = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        heads.append((float(np.array(ipp, float) @ nvec), f))
    heads.sort()
    idx = np.linspace(0, len(heads) - 1, n).round().astype(int)
    out = []
    for _, f in [heads[i] for i in idx]:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        if getattr(ds, "RescaleSlope", None) is not None:
            img = img * float(ds.RescaleSlope) + float(getattr(ds, "RescaleIntercept", 0))
        ps = float(ds.PixelSpacing[0])
        half = CROP_MM / 2 / ps
        cy, cx = img.shape[0] / 2, img.shape[1] / 2
        img = img[int(max(0, cy - half)):int(min(img.shape[0], cy + half)),
                  int(max(0, cx - half)):int(min(img.shape[1], cx + half))]
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-3), 0, 1)
        out.append(np.array(Image.fromarray((img * 255).astype(np.uint8))
                            .resize((SIZE, SIZE), Image.BILINEAR)))
    return np.stack(out)


class Net(nn.Module):
    """RSNA-winner aggregation: BiGRU over slice triplets + attention-MIL
    pooling per series, masked attention over the 4 series slots (replaces
    the plain means of the 0.915 baseline; encoder unchanged)."""

    def __init__(self):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnetv2_s", pretrained=False,
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


def preprocess_study(uid):
    """CPU part of predict_study: returns (vol, mask, sex) or None."""
    infos = [si for sd in sorted((COMP / SPLIT / uid).iterdir())
             if (si := series_info(sd)) is not None]
    vol = np.zeros((len(SLOTS), N_SLICES, SIZE, SIZE), np.uint8)
    mask = np.zeros(len(SLOTS), bool)
    for k, (plane, want_fs) in enumerate(SLOTS):
        cand = [i for i in infos if i["plane"] == plane and i["fatsat"] == want_fs] \
            or [i for i in infos if i["plane"] == plane]
        cand.sort(key=lambda i: (-i["n"], i["dir"].name))
        if cand:
            arr = load_series(cand[0]["dir"], N_SLICES)  # cache used all 24
            if arr is not None:
                vol[k], mask[k] = arr[:N_SLICES], True
    if not mask.any():
        return None
    sex = float(any(i["sex"] == "M" for i in infos))
    return vol, mask, sex


def forward_study(net, pre):
    if pre is None:
        return PRIOR
    vol, mask, sex = pre
    x = torch.from_numpy(vol).float().div_(255).unsqueeze(0).to(DEV)
    m = torch.from_numpy(mask).unsqueeze(0).to(DEV)
    s = torch.tensor([sex]).to(DEV)
    with torch.no_grad(), torch.amp.autocast(DEV if DEV == "cuda" else "cpu"):
        p = torch.sigmoid(net(x, m, s)).float().cpu().numpy()[0]
    return p.tolist()



def _prep_threaded(uid, n_threads=4):
    """preprocess_study with the per-slot series loads parallelised in threads."""
    from concurrent.futures import ThreadPoolExecutor
    infos = [si for sd in sorted((COMP / SPLIT / uid).iterdir())
             if (si := series_info(sd)) is not None]
    vol = np.zeros((len(SLOTS), N_SLICES, SIZE, SIZE), np.uint8)
    mask = np.zeros(len(SLOTS), bool)
    picks = []
    for k, (plane, want_fs) in enumerate(SLOTS):
        cand = [i for i in infos if i["plane"] == plane and i["fatsat"] == want_fs] \
            or [i for i in infos if i["plane"] == plane]
        cand.sort(key=lambda i: (-i["n"], i["dir"].name))
        if cand:
            picks.append((k, cand[0]["dir"]))
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for (k, _), arr in zip(picks, ex.map(lambda p: load_series(p[1], N_SLICES), picks)):
            if arr is not None:
                vol[k], mask[k] = arr[:N_SLICES], True
    if not mask.any():
        return None
    return vol, mask, float(any(i["sex"] == "M" for i in infos))


def _prep_t4(uid):
    return _prep_threaded(uid, 4)


def _prep_t8(uid):
    return _prep_threaded(uid, 8)


def main():
    """Concurrency pass at the current 4x24@320 config (cold cache, disjoint
    subsets): process-pool size x per-study series threads."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import os
    print("cpus", os.cpu_count(), flush=True)
    all_uids = sorted(pd.read_csv(COMP / "train.csv")["StudyInstanceUID"])
    variants = [("pool4", 4, preprocess_study, "proc"), ("pool8", 8, preprocess_study, "proc"),
                ("pool16", 16, preprocess_study, "proc"),
                ("pool4 x 4thr", 4, _prep_t4, "proc"), ("pool8 x 4thr", 8, _prep_t4, "proc"),
                ("pool4 x 8thr", 4, _prep_t8, "proc"),
                ("threads16 flat", 16, preprocess_study, "thread"),
                ("threads32 flat", 32, preprocess_study, "thread")]
    for vi, (name, nw, fn, kind) in enumerate(variants):
        uids = all_uids[600 + vi * N_STUDIES: 600 + (vi + 1) * N_STUDIES]
        Ex = ProcessPoolExecutor if kind == "proc" else ThreadPoolExecutor
        t0 = time.time()
        with Ex(max_workers=nw) as ex:
            n_ok = sum(p is not None for p in ex.map(fn, uids))
        dt = time.time() - t0
        print(f"{name:16s} {dt/len(uids):.3f} s/study -> {HIDDEN_N}: {dt/len(uids)*HIDDEN_N/60:.1f} min (ok {n_ok})", flush=True)


if __name__ == "__main__":
    main()
