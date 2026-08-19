"""DICOM-level data audit for RSNA Knee Abnormality Detection.

Runs on Kaggle (competition data mounted, CPU). Reads headers only — no pixels.
Outputs: series_meta.csv (one row per series) and audit.json (aggregates).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")
OUT = Path("/kaggle/working")

HDR_TAGS = [
    "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "RepetitionTime", "EchoTime", "ScanningSequence", "ScanOptions",
    "SeriesDescription", "Manufacturer", "ManufacturerModelName",
    "MagneticFieldStrength", "StationName", "DeviceSerialNumber",
    "InstitutionName", "PhotometricInterpretation", "BitsStored",
    "BodyPartExamined", "Laterality", "PatientSex", "PatientAge",
]


def plane_from_orientation(iop):
    # normal of the image plane; dominant axis => plane
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    n = np.abs(np.cross(r, c))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]


def read_series(series_dir):
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None
    mid = files[len(files) // 2]
    ds = pydicom.dcmread(mid, stop_before_pixels=True)
    row = {"SeriesInstanceUID": series_dir.name, "StudyInstanceUID": series_dir.parent.name,
           "n_slices": len(files)}
    for t in HDR_TAGS:
        v = getattr(ds, t, None)
        row[t] = str(v) if v is not None else None
    iop = getattr(ds, "ImageOrientationPatient", None)
    row["derived_plane"] = plane_from_orientation(iop) if iop is not None else None
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and getattr(ds, "Rows", None):
        row["fov_mm"] = round(float(ps[0]) * int(ds.Rows), 1)
    return row


def check_slice_order(series_dir):
    """True if filename sort order == spatial order along the slice normal."""
    files = sorted(series_dir.glob("*.dcm"))
    pos, iop = [], None
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True,
                             specific_tags=["ImagePositionPatient", "ImageOrientationPatient"])
        p = getattr(ds, "ImagePositionPatient", None)
        if p is None:
            return None
        if iop is None:
            iop = getattr(ds, "ImageOrientationPatient", None)
        pos.append(np.array(p, float))
    if iop is None or len(pos) < 3:
        return None
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    proj = [float(p @ n) for p in pos]
    return bool(np.all(np.diff(proj) > 0) or np.all(np.diff(proj) < 0))


def main():
    series_dirs = [d for study in sorted((ROOT / "train_series").iterdir())
                   for d in sorted(study.iterdir())]
    print(f"{len(series_dirs)} series dirs")
    rows = []
    for i, sd in enumerate(series_dirs):
        r = read_series(sd)
        if r:
            rows.append(r)
        if i % 2000 == 0:
            print(i, flush=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(OUT / "series_meta.csv", index=False)

    rng = np.random.default_rng(0)
    sample = rng.choice(len(series_dirs), size=min(300, len(series_dirs)), replace=False)
    order_ok = [check_slice_order(series_dirs[i]) for i in sample]
    order_ok = [o for o in order_ok if o is not None]

    csv_plane = pd.read_csv(ROOT / "train_series.csv").set_index("SeriesInstanceUID")["Anatomical_Plane"]
    merged = meta.dropna(subset=["derived_plane"]).set_index("SeriesInstanceUID")
    plane_agree = float((merged["derived_plane"] == csv_plane.reindex(merged.index)).mean())

    audit = {
        "n_series": len(meta),
        "n_studies": meta["StudyInstanceUID"].nunique(),
        "slices_per_series": {k: int(v) for k, v in
                              meta["n_slices"].describe()[["min", "25%", "50%", "75%", "max"]].items()},
        "filename_order_matches_spatial_frac": float(np.mean(order_ok)),
        "order_checked_n": len(order_ok),
        "csv_plane_agreement_with_geometry": plane_agree,
        "fov_mm_deciles": [round(x, 1) for x in
                           np.nanpercentile(meta["fov_mm"].astype(float), range(0, 101, 10))],
        "monochrome1_series": int((meta["PhotometricInterpretation"] == "MONOCHROME1").sum()),
        "manufacturers": meta["Manufacturer"].value_counts().to_dict(),
        "field_strengths": meta["MagneticFieldStrength"].value_counts().to_dict(),
        "n_scanners": int(meta["DeviceSerialNumber"].nunique()),
        "n_institutions": int(meta["InstitutionName"].nunique()),
        "laterality_counts": meta["Laterality"].value_counts(dropna=False).to_dict(),
        "patient_sex": meta.groupby("StudyInstanceUID")["PatientSex"].first().value_counts().to_dict(),
        "bits_stored": meta["BitsStored"].value_counts().to_dict(),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps(audit, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
