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



def _pos_order(files):
    heads = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=[
            "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"])
        iop, ipp = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "ImagePositionPatient", None)
        if iop is None or ipp is None:
            return None, None
        nvec = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        heads.append((float(np.array(ipp, float) @ nvec), int(getattr(ds, "InstanceNumber", -1)), f))
    heads.sort()
    return [f for _, _, f in heads], [i for _, i, _ in heads]


def _name_key(f):
    st = "".join(ch if ch.isdigit() else " " for ch in f.stem).split()
    return int(st[-1]) if st else 0


def load_series_byname(sdir, n):
    """Sample slices by filename order and open ONLY the sampled files."""
    files = sorted(Path(sdir).glob("*.dcm"), key=_name_key)
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    out = []
    for f in [files[i] for i in idx]:
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


def preprocess_byname(uid):
    infos = [si for sd in sorted((COMP / SPLIT / uid).iterdir())
             if (si := series_info(sd)) is not None]
    vol = np.zeros((len(SLOTS), N_SLICES, SIZE, SIZE), np.uint8)
    mask = np.zeros(len(SLOTS), bool)
    for k, (plane, want_fs) in enumerate(SLOTS):
        cand = [i for i in infos if i["plane"] == plane and i["fatsat"] == want_fs] \
            or [i for i in infos if i["plane"] == plane]
        cand.sort(key=lambda i: (-i["n"], i["dir"].name))
        if cand:
            arr = load_series_byname(cand[0]["dir"], N_SLICES)
            if arr is not None:
                vol[k], mask[k] = arr[:N_SLICES], True
    if not mask.any():
        return None
    return vol, mask, float(any(i["sex"] == "M" for i in infos))


def check_order(uid):
    """Per series: does filename order equal position order (or its reverse)?"""
    res = []
    for sd in sorted((COMP / SPLIT / uid).iterdir()):
        files = sorted(Path(sd).glob("*.dcm"), key=_name_key)
        if len(files) < 3:
            continue
        pos, inst = _pos_order(files)
        if pos is None:
            res.append(("nogeom", len(files))); continue
        same = pos == files or pos == files[::-1]
        inst_mono = inst == sorted(inst) or inst == sorted(inst, reverse=True)
        res.append(("match" if same else "mismatch", len(files), inst_mono, files[0].name))
    return res


def main():
    from concurrent.futures import ProcessPoolExecutor
    global N_SLICES, SIZE, SLOTS
    all_uids = sorted(pd.read_csv(COMP / "train.csv")["StudyInstanceUID"])
    # 1) how often does filename order == position order?
    tot = {"match": 0, "mismatch": 0, "nogeom": 0}; ex_names = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(check_order, all_uids[1200:1400]):
            for item in r:
                tot[item[0]] += 1
                if item[0] == "mismatch" and len(ex_names) < 5:
                    ex_names.append(item)
    print("filename-vs-position order over 200 studies:", tot, "examples:", ex_names, flush=True)
    # 2) cost: pool8, position-ordered (current) vs filename-ordered (only sampled files opened)
    S4 = [("Sagittal", True), ("Sagittal", False), ("Coronal", True), ("Axial", True)]
    S6 = [("Sagittal", True), ("Sagittal", False), ("Coronal", True),
          ("Coronal", False), ("Axial", True), ("Axial", False)]
    variants = [("4x24 pos", S4, 24, preprocess_study), ("4x24 byname", S4, 24, preprocess_byname),
                ("4x12 byname", S4, 12, preprocess_byname), ("6x16 byname", S6, 16, preprocess_byname),
                ("4x16 byname", S4, 16, preprocess_byname), ("4x12 pos", S4, 12, preprocess_study)]
    for vi, (name, slots, ns, fn) in enumerate(variants):
        SLOTS, N_SLICES, SIZE = slots, ns, 320
        uids = all_uids[1500 + vi * N_STUDIES: 1500 + (vi + 1) * N_STUDIES]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=8) as ex:
            n_ok = sum(p is not None for p in ex.map(fn, uids))
        dt = time.time() - t0
        print(f"{name:14s} pool8 {dt/len(uids):.3f} s/study -> {HIDDEN_N}: {dt/len(uids)*HIDDEN_N/60:.1f} min (ok {n_ok})", flush=True)


if __name__ == "__main__":
    main()
