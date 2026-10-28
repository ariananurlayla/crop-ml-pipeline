"""
NASA CropHarvest loader + adapter (Tseng et al., NeurIPS D&B 2021).

Downloads the two files the pipeline needs from the official Zenodo record
(v14, id 10251170) and converts the per-point HDF5 arrays into the tabular
layout the rest of the pipeline already consumes:

  features.tar.gz  (~82 MB) -> features/arrays/{index}_{dataset}.h5, key
                               "array", shape (12 months, 18 bands) float32
  labels.geojson   (~86 MB) -> per-row lat/lon, source dataset, crop label

The ``cropharvest`` pip package is deliberately NOT used: its pins
(pandas<2.0, geopandas==0.9.0, rasterio, fiona) conflict with this pipeline
and need a conda env on Windows. Only ``h5py`` is required on top of the
existing requirements.

The 18 bands (band-major after flattening, see ``build_dataframe``):
VV, VH, B2..B8A, B9, B11, B12, temperature_2m, total_precipitation,
elevation, slope, NDVI  ->  216 columns "{band}_{month}".

Labels are mapped onto the pipeline's 4-class schema (``config.py``):
rice -> 0, maize -> 1, sugarcane -> 2, other crops (is_crop=True) -> 3.
Rows with ``is_test=True`` are excluded (their arrays live in
test_features.tar.gz, which this pipeline does not use) as are the
binary-only rows whose ``classification_label`` is null.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from config import (CROPHARVEST_CHANNELS, CROPHARVEST_FILES,
                    CROPHARVEST_LABEL_MAP, CROPHARVEST_ZENODO_ID, MONTHS)

try:
    import h5py
except ImportError:   # only needed to read the feature arrays
    h5py = None

META_COLS = ["cropharvest_index", "latitude", "longitude", "source_dataset"]
RAW_LABEL_COL = "raw_label"
ARRAY_SHAPE = (12, 18)
DEFAULT_OTHER_SAMPLES = 4000
# spatial-blocking cell size for the grouped split (~111 km at the equator)
SPATIAL_CELL_DEG = 1.0
_EXTRACT_OK = "_EXTRACT_OK"

_ZENODO_API = f"https://zenodo.org/api/records/{CROPHARVEST_ZENODO_ID}"
_HARDCODED_URLS = {
    name: f"https://zenodo.org/record/{CROPHARVEST_ZENODO_ID}/files/{fname}?download=1"
    for name, fname in CROPHARVEST_FILES.items()
}
_LABEL_LOOKUP = {k.strip().casefold(): v for k, v in CROPHARVEST_LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# download + extraction (idempotent)
# ---------------------------------------------------------------------------

def _zenodo_file_info() -> dict:
    """Resolve ``{name: {"url", "size", "md5"}}`` from the Zenodo API, falling
    back to hardcoded ``?download=1`` URLs when the API is unreachable."""
    info = {}
    try:
        with urllib.request.urlopen(_ZENODO_API, timeout=30) as resp:
            record = json.load(resp)
        for f in record.get("files", []):
            key = f.get("key", "")
            for name, fname in CROPHARVEST_FILES.items():
                if key == fname:
                    info[name] = {
                        "url": f["links"]["self"],
                        "size": f.get("size"),
                        "md5": f.get("checksum", "").removeprefix("md5:"),
                    }
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        print(f"[cropharvest] Zenodo API unreachable ({exc}); using hardcoded URLs")
    for name, url in _HARDCODED_URLS.items():
        info.setdefault(name, {"url": url, "size": None, "md5": None})
    return info


def download_file(url: str, dest: str, expected_size: int = None,
                  expected_md5: str = None, chunk: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest + ".part"``, then atomically rename into place.

    An interrupted download can therefore never be mistaken for a complete
    file - skip-if-exists only ever sees fully downloaded, verified files.
    """
    part = dest + ".part"
    downloaded = 0
    md5 = hashlib.md5()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0) or expected_size or 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                md5.update(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"\r[cropharvest] downloading {os.path.basename(dest)} "
                          f"{downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)",
                          end="", flush=True)
        print()
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    if expected_size and downloaded != expected_size:
        os.remove(part)
        raise RuntimeError(f"[cropharvest] size mismatch for {os.path.basename(dest)}: "
                           f"got {downloaded} bytes, expected {expected_size}")
    if expected_md5 and md5.hexdigest() != expected_md5:
        os.remove(part)
        raise RuntimeError(f"[cropharvest] md5 mismatch for {os.path.basename(dest)} - "
                           "delete the file and retry")
    os.replace(part, dest)


def _safe_extract_tar(tar_path: str, root: str, features_dir: str) -> None:
    """Extract features.tar.gz (untrusted input) into ``features_dir``.

    Path-traversal guarded (every member must resolve inside the extraction
    directory), extracted to a temp dir that is moved into place atomically,
    then an ``_EXTRACT_OK`` marker makes interrupted extractions resume
    cleanly instead of re-doing the work.
    """
    tmp = os.path.join(root, ".extract_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    tmp_abs = os.path.abspath(tmp)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.startswith("features/")]
        print(f"[cropharvest] extracting {len(members)} feature files "
              "(one-time, a few minutes)...")
        for i, member in enumerate(members, 1):
            rel = os.path.relpath(member.name, "features")
            target = os.path.abspath(os.path.join(tmp, rel))
            if not target.startswith(tmp_abs + os.sep):
                raise RuntimeError(f"[cropharvest] unsafe path in archive: {member.name}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tar.extractfile(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if i % 5000 == 0:
                print(f"[cropharvest]   {i}/{len(members)} files")
    if os.path.exists(features_dir):
        shutil.rmtree(features_dir)
    os.replace(tmp, features_dir)
    with open(os.path.join(features_dir, _EXTRACT_OK), "w", encoding="utf-8") as f:
        f.write("ok\n")


def ensure_cropharvest(root: str, download: bool = True,
                       include_features: bool = True) -> str:
    """Make sure the CropHarvest files exist under ``root`` (downloading and
    extracting on first use). Idempotent: completed downloads and an
    ``_EXTRACT_OK`` marker are never repeated. Returns the absolute root."""
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    info = _zenodo_file_info()
    names = ("labels", "features") if include_features else ("labels",)

    for name in names:
        fname = CROPHARVEST_FILES[name]
        dest = os.path.join(root, fname)
        if os.path.exists(dest):
            continue
        if not download:
            raise RuntimeError(
                f"[cropharvest] {fname} missing under {root} and --no-download "
                "was given - run once without --no-download to fetch it")
        download_file(info[name]["url"], dest,
                      expected_size=info[name]["size"],
                      expected_md5=info[name]["md5"])

    if include_features:
        features_dir = os.path.join(root, "features")
        if not os.path.exists(os.path.join(features_dir, _EXTRACT_OK)):
            _safe_extract_tar(os.path.join(root, CROPHARVEST_FILES["features"]),
                              root, features_dir)
    return root


# ---------------------------------------------------------------------------
# labels / arrays / mapping
# ---------------------------------------------------------------------------

def load_labels(root: str) -> pd.DataFrame:
    """Parse labels.geojson into a plain DataFrame (``properties`` only)."""
    path = os.path.join(root, CROPHARVEST_FILES["labels"])
    print("[cropharvest] parsing labels.geojson ...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    except MemoryError:
        raise MemoryError("[cropharvest] labels.geojson is too large to parse with "
                          "json.load - close other programs and retry") from None
    rows = []
    for feat in gj["features"]:
        p = feat.get("properties") or {}
        lat, lon = p.get("lat"), p.get("lon")
        rows.append({
            "cropharvest_index": int(p["index"]),
            "dataset": str(p["dataset"]),
            "is_crop": bool(p.get("is_crop") or False),
            "is_test": bool(p.get("is_test") or False),
            "classification_label": p.get("classification_label"),
            "label": p.get("label"),
            "latitude": float(lat) if lat is not None else np.nan,
            "longitude": float(lon) if lon is not None else np.nan,
        })
    return pd.DataFrame(rows)


def load_features_array(root: str, index: int, dataset: str) -> np.ndarray:
    """Read one per-point array ``features/arrays/{index}_{dataset}.h5``."""
    if h5py is None:
        sys.exit("[cropharvest] h5py is required to read the feature arrays - "
                 "run: pip install h5py")
    path = os.path.join(root, "features", "arrays", f"{index}_{dataset}.h5")
    with h5py.File(path, "r") as f:
        x = np.asarray(f["array"], dtype=np.float32)
    if x.shape != ARRAY_SHAPE:
        raise ValueError(f"[cropharvest] unexpected array shape {x.shape} in {path}")
    return x


def map_label(raw) -> int:
    """Map a raw CropHarvest label string to the pipeline class (-1 unmapped).

    Falls back to the part before " - ": French tenure qualifiers such as
    "canne à sucre - fermage" (sugarcane under leasehold) are still
    sugarcane.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return -1
    norm = str(raw).strip().casefold()
    mapped = _LABEL_LOOKUP.get(norm)
    if mapped is not None:
        return mapped
    return _LABEL_LOOKUP.get(norm.split(" - ")[0].strip(), -1)


def spatial_grid_groups(df: pd.DataFrame, cell_deg: float = SPATIAL_CELL_DEG):
    """Group ids from rounded lat/lon cells - the spatial blocking for the
    grouped split. Samples in the same ~1-degree cell share geography, so a
    GroupShuffleSplit on these ids keeps nearby points in one fold.

    (Grouping by source dataset is a worse proxy: several CropHarvest source
    datasets, e.g. JECAM, span multiple continents.)
    """
    lat = (df["latitude"] / cell_deg).round().astype(int)
    lon = (df["longitude"] / cell_deg).round().astype(int)
    return lat.astype(str) + "_" + lon.astype(str)


def filter_region(df: pd.DataFrame, bbox: dict) -> pd.DataFrame:
    """Keep only the rows inside a lat/lon bounding box (``--region``)."""
    m = (df["latitude"].between(bbox["lat_min"], bbox["lat_max"])
         & df["longitude"].between(bbox["lon_min"], bbox["lon_max"]))
    return df[m].reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataFrame assembly (the 216-feature table the pipeline consumes)
# ---------------------------------------------------------------------------

def build_dataframe(root: str, other_samples: int = DEFAULT_OTHER_SAMPLES,
                    seed: int = 42, force: bool = False) -> pd.DataFrame:
    """Build the 216-feature + ``crop_class`` CropHarvest table, cached as CSV.

    The cache filename encodes the sampling parameters, so a different
    ``other_samples``/``seed`` never silently reuses a stale cache.
    Meta columns (index/lat/lon/source dataset) are kept for the spatial
    split and the split-map figure; dataset.py excludes them from features
    via its ``ignore_cols`` argument.
    """
    if other_samples < 1:
        raise ValueError("[cropharvest] --other-samples must be >= 1")
    cache = os.path.join(root, f"cropharvest_features_o{other_samples}_s{seed}.csv")
    if os.path.exists(cache) and not force:
        print(f"[cropharvest] cache hit: {cache}")
        return pd.read_csv(cache)

    labels = load_labels(root)
    n_raw = len(labels)
    labels = labels[~labels["is_test"]].copy()   # arrays live in test_features/, not here
    n_test = n_raw - len(labels)
    labels = labels[labels["classification_label"].notna()].copy()  # binary-only rows
    n_null = n_raw - n_test - len(labels)
    print(f"[cropharvest] labels: {n_raw} total | excluded {n_test} is_test + "
          f"{n_null} null-classification-label")

    labels["mapped"] = labels["label"].map(map_label)
    unmapped = labels.loc[(labels["mapped"] == -1) & labels["is_crop"], "label"]
    print("[cropharvest] top unmapped crop labels (check config.CROPHARVEST_LABEL_MAP):")
    for lab, cnt in unmapped.value_counts().head(20).items():
        print(f"    {lab}: {cnt}")

    focus = labels[labels["mapped"] >= 0].copy()
    focus["crop_class"] = focus["mapped"].astype(int)
    other_pool = labels[(labels["mapped"] == -1) & labels["is_crop"]]
    n_other = min(other_samples, len(other_pool))
    if n_other < other_samples:
        print(f"[cropharvest] warning: only {n_other} Other rows available "
              f"(asked {other_samples})")
    other = other_pool.sample(n=n_other, random_state=seed).copy()
    other["crop_class"] = 3
    keep = pd.concat([focus, other], ignore_index=True)

    print(f"[cropharvest] building arrays for {len(keep)} rows ...")
    x = np.empty((len(keep), ARRAY_SHAPE[0], ARRAY_SHAPE[1]), dtype=np.float32)
    missing = []
    for i, row in enumerate(keep.itertuples()):
        if i % 1000 == 0:
            print(f"[cropharvest]   {i}/{len(keep)}")
        try:
            x[i] = load_features_array(root, row.cropharvest_index, row.dataset)
        except (OSError, KeyError, ValueError) as exc:
            missing.append(i)
    if missing:
        keep = keep.drop(index=missing).reset_index(drop=True)
        x = np.delete(x, missing, axis=0)
        print(f"[cropharvest] warning: dropped {len(missing)} rows with "
              "missing feature files")

    # band-major flatten: h5 is (12 months, 18 bands); transpose so columns
    # come out as all 18 bands of Jan, then Feb, ... -> "{band}_{month}"
    x = x.transpose(0, 2, 1).reshape(len(keep), -1)
    cols = [f"{band}_{month}" for band in CROPHARVEST_CHANNELS for month in MONTHS]
    out = pd.DataFrame(x, columns=cols)
    out["crop_class"] = keep["crop_class"].to_numpy()
    out[RAW_LABEL_COL] = keep["label"].to_numpy()
    out["cropharvest_index"] = keep["cropharvest_index"].to_numpy()
    out["latitude"] = keep["latitude"].to_numpy()
    out["longitude"] = keep["longitude"].to_numpy()
    out["source_dataset"] = keep["dataset"].to_numpy()

    out.to_csv(cache, index=False, float_format="%.9g")   # 9 sig. digits: exact float32 round-trip
    counts = out["crop_class"].value_counts().sort_index().to_dict()
    print(f"[cropharvest] wrote {len(out)} rows x {out.shape[1]} cols -> {cache}")
    print(f"[cropharvest] class counts (0=Rice 1=Corn 2=Sugarcane 3=Other): {counts}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CropHarvest data tool (Zenodo record "
                                f"{CROPHARVEST_ZENODO_ID})")
    p.add_argument("--root", default="data/cropharvest",
                   help="directory for the CropHarvest files")
    p.add_argument("--labels-only", action="store_true",
                   help="download/parse labels only (no feature arrays)")
    p.add_argument("--other-samples", type=int, default=DEFAULT_OTHER_SAMPLES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="rebuild the cached CSV")
    args = p.parse_args()

    ensure_cropharvest(args.root, include_features=not args.labels_only)
    labels = load_labels(args.root)
    keep = labels[~labels["is_test"] & labels["classification_label"].notna()]
    mapped = keep["label"].map(map_label)
    other_crops = int(keep.loc[mapped == -1, "is_crop"].sum())
    print(f"[cropharvest] mapped labels: rice {int((mapped == 0).sum())} | "
          f"maize {int((mapped == 1).sum())} | sugarcane {int((mapped == 2).sum())} | "
          f"other crops {other_crops}")
    print(f"[cropharvest] excluded: is_test {int(labels['is_test'].sum())} | "
          f"null-classification-label "
          f"{int(labels['classification_label'].isna().sum())} | "
          f"non_crop {int((labels['is_crop'] == False).sum())}")
    if not args.labels_only:
        build_dataframe(args.root, other_samples=args.other_samples,
                        seed=args.seed, force=args.force)
