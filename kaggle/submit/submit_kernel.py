"""Submission kernel (offline, T4). Produces /kaggle/working/submission.csv.

Mirrors the training preprocessing exactly (position-sorted slices, 140mm
center crop, per-slice percentile norm, 4 slots with header-derived fat-sat)
and runs the fold-0 EfficientNetV2-S checkpoint from the train kernel output.
Any study that fails preprocessing gets the training-prevalence prior so the
submission always has every required row.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from PIL import Image

N_SLICES = 15
SIZE = 256
CROP_MM = 140.0
SLOTS = [("Sagittal", True), ("Sagittal", False), ("Coronal", True), ("Axial", True)]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PRIOR = [0.30, 0.12, 0.40, 0.30, 0.25, 0.18, 0.30, 0.45, 0.35, 0.18, 0.25, 0.20]
FS_PAT = re.compile(r"(?i)\bfs\b|fat.?sat|spair|spir|stir|tirm|_fs|fs_|fatsat")

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/test_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/test_series.csv")))[0]
CKPT = sorted(INPUT.rglob("effv2s_f0.pt"))[0]
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
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnetv2_s", pretrained=False,
                                     num_classes=0, in_chans=3)
        d = self.enc.num_features
        self.head = nn.Sequential(nn.Linear(d + 1, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, len(LABELS)))

    def forward(self, x, mask, sex):
        b, k, n, h, w = x.shape
        trip = x.view(b * k, n // 3, 3, h, w).flatten(0, 1)
        f = self.enc(trip).view(b, k, n // 3, -1).mean(2)
        m = mask.float().unsqueeze(-1)
        emb = (f * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([emb, sex.unsqueeze(-1)], -1))


def predict_study(net, uid):
    infos = [si for sd in sorted((COMP / "test_series" / uid).iterdir())
             if (si := series_info(sd)) is not None]
    vol = np.zeros((len(SLOTS), N_SLICES, SIZE, SIZE), np.uint8)
    mask = np.zeros(len(SLOTS), bool)
    for k, (plane, want_fs) in enumerate(SLOTS):
        cand = [i for i in infos if i["plane"] == plane and i["fatsat"] == want_fs] \
            or [i for i in infos if i["plane"] == plane]
        cand.sort(key=lambda i: (-i["n"], i["dir"].name))
        if cand:
            arr = load_series(cand[0]["dir"], N_SLICES)
            if arr is not None:
                vol[k], mask[k] = arr, True
    if not mask.any():
        return PRIOR
    sex = float(any(i["sex"] == "M" for i in infos))
    x = torch.from_numpy(vol).float().div_(255).unsqueeze(0).to(DEV)
    m = torch.from_numpy(mask).unsqueeze(0).to(DEV)
    s = torch.tensor([sex]).to(DEV)
    with torch.no_grad(), torch.amp.autocast(DEV if DEV == "cuda" else "cpu"):
        p = torch.sigmoid(net(x, m, s)).float().cpu().numpy()[0]
    return p.tolist()


def main():
    test = pd.read_csv(COMP / "test.csv")
    net = Net().to(DEV)
    net.load_state_dict(torch.load(CKPT, map_location=DEV))
    net.eval()
    rows = []
    for i, uid in enumerate(test["StudyInstanceUID"]):
        try:
            p = predict_study(net, uid)
        except Exception as e:
            print(uid, "FAIL", e, flush=True)
            p = PRIOR
        rows.append([uid] + list(p))
        if i % 100 == 0:
            print(i, flush=True)
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print("wrote", len(sub), "rows")


if __name__ == "__main__":
    main()
