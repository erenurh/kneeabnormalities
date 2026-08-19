"""Build the cached training arrays (CPU kernel).

Per study, selects up to 4 series slots (Sagittal-FS, Sagittal-nonFS,
Coronal-FS, Axial-FS), reads N_SLICES evenly spaced position-sorted slices,
crops a fixed CROP_MM window around the image center, resizes to SIZE and
stores uint8 percentile-normalized volumes: one {uid}.npz per study with a
(4, N_SLICES, SIZE, SIZE) array + slot-present mask.

Fat suppression is derived from headers (ScanOptions/SeriesDescription),
not the buggy CSV columns. Output feeds the training kernel as a dataset.
"""
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

N_SLICES = 12
SIZE = 256
CROP_MM = 140.0
SLOTS = [("Sagittal", True), ("Sagittal", False), ("Coronal", True), ("Axial", True)]

INPUT = Path("/kaggle/input")
COMP = (sorted(p.parent for p in INPUT.glob("*/train_series.csv"))
        or sorted(p.parent for p in INPUT.glob("*/*/train_series.csv")))[0]
META = sorted(INPUT.rglob("series_meta.csv"))[0]
OUT = Path("/kaggle/working/cache")
OUT.mkdir(parents=True, exist_ok=True)

FS_PAT = re.compile(r"(?i)\bfs\b|fat.?sat|spair|spir|stir|tirm|_fs|fs_|fatsat")


def is_fatsat(row):
    return bool(FS_PAT.search(f"{row.ScanOptions} {row.SeriesDescription}"))


def load_series(sdir, n):
    files = list(Path(sdir).glob("*.dcm"))
    heads = []
    for f in files:
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
        y0, y1 = int(max(0, cy - half)), int(min(img.shape[0], cy + half))
        x0, x1 = int(max(0, cx - half)), int(min(img.shape[1], cx + half))
        img = img[y0:y1, x0:x1]
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-3), 0, 1)
        out.append(np.array(Image.fromarray((img * 255).astype(np.uint8))
                            .resize((SIZE, SIZE), Image.BILINEAR)))
    return np.stack(out)


def pick_series(g):
    """g: series_meta rows of one study -> series dir name per slot (or None)."""
    chosen = []
    for plane, want_fs in SLOTS:
        cand = g[(g["derived_plane"] == plane) & (g["fatsat"] == want_fs)]
        if not len(cand):
            cand = g[g["derived_plane"] == plane]
        # most slices first; deterministic tiebreak by UID
        cand = cand.sort_values(["n_slices", "SeriesInstanceUID"], ascending=[False, True])
        chosen.append(cand["SeriesInstanceUID"].iloc[0] if len(cand) else None)
    return chosen


def process_study(args):
    uid, series_uids = args
    vol = np.zeros((len(SLOTS), N_SLICES, SIZE, SIZE), np.uint8)
    mask = np.zeros(len(SLOTS), bool)
    for k, suid in enumerate(series_uids):
        if suid is None:
            continue
        try:
            arr = load_series(COMP / "train_series" / uid / suid, N_SLICES)
            if arr is not None:
                vol[k], mask[k] = arr, True
        except Exception as e:
            print(uid, suid, "FAIL", e, flush=True)
    np.savez_compressed(OUT / f"{uid}.npz", vol=vol, mask=mask)
    return uid


def main():
    meta = pd.read_csv(META)
    meta["fatsat"] = meta.apply(is_fatsat, axis=1)
    jobs = [(uid, pick_series(g)) for uid, g in meta.groupby("StudyInstanceUID")]
    print(len(jobs), "studies")
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, _ in enumerate(ex.map(process_study, jobs, chunksize=8)):
            if i % 200 == 0:
                print(i, flush=True)
    print("done", len(list(OUT.glob("*.npz"))))


if __name__ == "__main__":
    main()
