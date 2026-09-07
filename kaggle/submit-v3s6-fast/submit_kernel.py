"""Submission kernel (offline, T4, fp16). Produces /kaggle/working/submission.csv.

Efficiency-entry version of submit-v2: identical preprocessing + v2 checkpoint,
but DICOM preprocessing runs in an 8-process pool feeding the GPU, and slices
are ordered by a raw-byte InstanceNumber scan (direction fixed from the first/
last file's ImagePositionPatient; per-series pydicom fallback) so only the
sampled files are fully parsed. Verified bit-identical to position sorting on
120 training studies (exp-43); ~7 min at hidden-test size vs ~17 min before. Measured on
the runtime harness (100 train studies, T4): sequential 2.15 s/study (~47 min
at 1,300 studies) vs pooled 0.40 s/study (~9 min); GPU forward is 0.15 s.

Mirrors the training preprocessing exactly (position-sorted slices, 140mm
center crop, per-slice percentile norm, 4 slots with header-derived fat-sat)
and runs the v3-s6 checkpoint (6 slots FS/non-FS x 15 slices, per-finding query attention, train-v3-s6).
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

N_SLICES = 15
N_SLOTS = 6
SIZE = 320
CROP_MM = 140.0
SLOTS = [("Sagittal", True), ("Sagittal", False), ("Coronal", True),
         ("Coronal", False), ("Axial", True), ("Axial", False)]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PRIOR = [0.30, 0.12, 0.40, 0.30, 0.25, 0.18, 0.30, 0.45, 0.35, 0.18, 0.25, 0.20]
FS_PAT = re.compile(r"(?i)\bfs\b|fat.?sat|spair|spir|stir|tirm|_fs|fs_|fatsat")

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/test_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/test_series.csv")))[0]
CKPT = sorted(INPUT.rglob("effv2s_v3s6_all.pt"))[0]
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


_TAG_IN = b"\x20\x00\x13\x00"   # (0020,0013) InstanceNumber, little endian


def fast_instance_number(f, nbytes=6144):
    """Read InstanceNumber from the raw header bytes (explicit or implicit VR
    little endian). Returns None if not found -> caller falls back to pydicom."""
    with open(f, "rb") as fh:
        buf = fh.read(nbytes)
    i = buf.find(_TAG_IN, 128)
    while i >= 0:
        vr = buf[i + 4:i + 6]
        if vr == b"IS":
            ln = int.from_bytes(buf[i + 6:i + 8], "little"); val = buf[i + 8:i + 8 + ln]
        else:  # implicit VR
            ln = int.from_bytes(buf[i + 4:i + 8], "little"); val = buf[i + 8:i + 8 + ln]
        try:
            if 0 < ln <= 12:
                return int(val.decode("ascii").strip("\x00 "))
        except ValueError:
            pass
        i = buf.find(_TAG_IN, i + 4)
    return None


def order_fast(files):
    """files sorted by InstanceNumber via the raw scan; None on any failure."""
    nums = [fast_instance_number(f) for f in files]
    if any(n is None for n in nums) or len(set(nums)) != len(nums):
        return None
    return [f for _, f in sorted(zip(nums, files))]


def _proj(f):
    ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=[
        "ImagePositionPatient", "ImageOrientationPatient"])
    iop, ipp = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        return None
    nvec = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return float(np.array(ipp, float) @ nvec)


def load_series_fast(sdir, n):
    files = list(Path(sdir).glob("*.dcm"))
    ordered = order_fast(files)
    if ordered is None:
        return load_series(sdir, n)  # pydicom fallback (position sort)
    a, b = _proj(ordered[0]), _proj(ordered[-1])
    if a is None or b is None:
        return load_series(sdir, n)
    if a > b:  # InstanceNumber runs against the geometric normal: flip
        ordered = ordered[::-1]
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    out = []
    for f in [ordered[i] for i in idx]:
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


def preprocess_study(uid):
    """CPU part: returns (vol, mask, sex) or None (-> PRIOR)."""
    infos = [si for sd in sorted((COMP / "test_series" / uid).iterdir())
             if (si := series_info(sd)) is not None]
    vol = np.zeros((len(SLOTS), N_SLICES, SIZE, SIZE), np.uint8)
    mask = np.zeros(len(SLOTS), bool)
    used = set()
    for k, (plane, want_fs) in enumerate(SLOTS):
        free = [i for i in infos if i["dir"].name not in used]
        cand = [i for i in free if i["plane"] == plane and i["fatsat"] == want_fs]
        if not cand and want_fs:
            cand = [i for i in free if i["plane"] == plane]
        cand.sort(key=lambda i: (-i["n"], i["dir"].name))
        if cand:
            used.add(cand[0]["dir"].name)
            arr = load_series_fast(cand[0]["dir"], N_SLICES)
            if arr is not None:
                vol[k], mask[k] = arr[:N_SLICES], True
    if not mask.any():
        return None
    sex = float(any(i["sex"] == "M" for i in infos))
    return vol, mask, sex


def safe_preprocess(uid):
    try:
        return preprocess_study(uid)
    except Exception as e:  # never let one study kill the pool
        print(uid, "FAIL", e, flush=True)
        return None


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



def main():
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    test = pd.read_csv(COMP / "test.csv")
    uids = list(test["StudyInstanceUID"])
    net = Net().to(DEV)
    net.load_state_dict(torch.load(CKPT, map_location=DEV))
    net.eval()
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (uid, pre) in enumerate(zip(uids, ex.map(safe_preprocess, uids))):
            try:
                p = forward_study(net, pre)
            except Exception as e:
                print(uid, "FAIL fwd", e, flush=True)
                p = PRIOR
            rows.append([uid] + list(p))
            if i % 100 == 0:
                el = time.time() - t0
                print(f"{i} elapsed={el:.0f}s per_study={el/max(i,1):.2f}s", flush=True)
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    assert len(sub) == len(test)
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"wrote {len(sub)} rows total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
