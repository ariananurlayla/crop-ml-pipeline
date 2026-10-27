"""
Synthetic data generator matching the 181-column IFRI schema.

Used to smoke-test the pipeline end-to-end before the real GEE export is
available. Class profiles are crude but phenologically plausible:

  * Rice       : flooded early (high NDWI, low VV backscatter), NDVI rises
                 through the cycle; days_after_planting shifts the cycle phase.
  * Corn       : two short cycles, no flood signal.
  * Sugarcane  : long slow NDVI ramp with a wide plateau.
  * Other      : irregular mixed profile (vegetables etc.).

``days_after_planting`` is only defined for Rice rows (NaN elsewhere),
mirroring the IFRI use case where flood detection targets paddy. The
regression head therefore trains on the rice subset and predicts HST.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MONTHS, S2_BANDS

STAT_NAMES = ["Mean", "Median", "StdDev", "Min", "Max",
              "P10", "P25", "P75", "P90", "Range", "Entropy"]

# Wet season Dec-Mar (monsoon Java): high rainfall
_RAIN_TEMPLATE = np.array([280, 260, 220, 150, 90, 50, 30, 25, 40, 80, 150, 240],
                          dtype=np.float64)


def _rice_ndvi(month_idx, dap):
    """NDVI of a 120-day rice cycle planted `dap` days ago."""
    age = (month_idx * 30.0 - dap) % 120.0
    if age < 15:
        return 0.18                                  # flooded / transplant
    if age < 55:
        return 0.18 + 0.55 * (age - 15) / 40.0       # vegetative ramp
    if age < 95:
        return 0.73                                  # generative plateau
    return max(0.20, 0.73 - 0.55 * (age - 95) / 25.0)  # ripening -> harvest


_CORN_NDVI = np.array([0.15, 0.25, 0.55, 0.78, 0.70, 0.45, 0.20, 0.15,
                       0.20, 0.50, 0.72, 0.40], dtype=np.float64)
_SUGARCANE_NDVI = np.array([0.35, 0.45, 0.55, 0.65, 0.72, 0.75, 0.75, 0.72,
                            0.68, 0.60, 0.50, 0.40], dtype=np.float64)


def _series_entropy(values):
    counts, _ = np.histogram(values, bins=5)
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def generate_synthetic_csv(path: str, n: int = 2000, seed: int = 0) -> str:
    """Write a synthetic CSV with the exact 181-column schema and return `path`."""
    rng = np.random.default_rng(seed)
    class_probs = [0.45, 0.20, 0.15, 0.20]  # Rice, Corn, Sugarcane, Other
    months = np.arange(12, dtype=np.float64)
    records = []

    for _ in range(n):
        cls = int(rng.choice(4, p=class_probs))
        # ~8% label noise: features come from the true class but the written
        # label is flipped (mimics mislabeled on-screen interpretation points),
        # which caps achievable accuracy at a realistic level
        label = cls
        if rng.random() < 0.08:
            label = int(rng.choice([c for c in range(4) if c != cls]))
        dap = np.nan
        # Wide amplitude jitter + noise so classes overlap realistically
        # (small rice fields / mixed pixels make corn vs sugarcane genuinely hard)
        amp = float(rng.uniform(0.70, 1.30))
        noise = float(rng.uniform(0.05, 0.12))

        if cls == 0:
            dap = float(rng.uniform(8, 115))
            ndvi = np.array([_rice_ndvi(m, dap) for m in months]) * amp
            ndvi += rng.normal(0, noise, 12)
        elif cls == 1:
            ndvi = _CORN_NDVI * amp + rng.normal(0, noise, 12)
        elif cls == 2:
            ndvi = _SUGARCANE_NDVI * amp + rng.normal(0, noise, 12)
        else:
            phase = rng.uniform(0, 2 * np.pi)
            ndvi = 0.30 + 0.20 * np.sin(months * 1.7 + phase) + rng.normal(0, noise, 12)
        ndvi = np.clip(ndvi, 0.05, 0.95)

        # NDWI: high only during the flooded transplant stage of rice
        if cls == 0:
            age = (months * 30.0 - dap) % 120.0
            ndwi = np.where(age < 15, 0.45, 0.08) + rng.normal(0, 0.05, 12)
        else:
            ndwi = 0.06 + rng.normal(0, 0.04, 12)
        ndwi = np.clip(ndwi, -0.1, 0.6)

        # Sentinel-2 bands derived from NDVI (spectral plausibility only)
        s2 = {}
        s2["B2"] = 0.12 - 0.04 * ndvi
        s2["B3"] = 0.15 - 0.05 * ndvi
        s2["B4"] = 0.25 - 0.15 * ndvi
        s2["B5"] = 0.08 + 0.50 * ndvi
        s2["B6"] = 0.05 + 0.40 * ndvi
        s2["B7"] = 0.04 + 0.30 * ndvi
        s2["B8"] = 0.10 + 0.60 * ndvi
        s2["B8A"] = 0.10 + 0.55 * ndvi

        # Sentinel-1: flooded rice paddies -> specular reflection -> very low VV
        vv_noise = 1.3
        if cls == 0:
            age = (months * 30.0 - dap) % 120.0
            vv = np.where(age < 15, -18.0, -13.0) + rng.normal(0, vv_noise, 12)
        elif cls == 1:
            vv = -11.0 + rng.normal(0, vv_noise, 12)
        elif cls == 2:
            vv = -10.0 + rng.normal(0, vv_noise, 12)
        else:
            vv = -12.0 + rng.normal(0, 1.5, 12)
        vh = vv - 5.5 + rng.normal(0, 0.8, 12)

        rain = _RAIN_TEMPLATE * rng.lognormal(0, 0.15, 12)

        # --- assemble one row ---
        row = {}
        for band, values in s2.items():
            for m, month in enumerate(MONTHS):
                row[f"S2_{band}_{month}"] = float(values[m] + rng.normal(0, 0.02))
        for m, month in enumerate(MONTHS):
            row[f"NDVI_{month}"] = float(ndvi[m])
            row[f"NDWI_{month}"] = float(ndwi[m])
            row[f"S1_VV_{month}"] = float(vv[m])
            row[f"S1_VH_{month}"] = float(vh[m])
            row[f"Rain_{month}"] = float(rain[m])

        row["Terrain_Elevation"] = float(rng.uniform(0, 2500))
        row["Terrain_Slope"] = float(rng.uniform(0, 25))
        row["Terrain_Aspect"] = float(rng.uniform(0, 360))

        for prefix, series in (("VV", vv), ("VH", vh)):
            row[f"{prefix}_Mean"] = float(series.mean())
            row[f"{prefix}_Median"] = float(np.median(series))
            row[f"{prefix}_StdDev"] = float(series.std())
            row[f"{prefix}_Min"] = float(series.min())
            row[f"{prefix}_Max"] = float(series.max())
            row[f"{prefix}_P10"] = float(np.percentile(series, 10))
            row[f"{prefix}_P25"] = float(np.percentile(series, 25))
            row[f"{prefix}_P75"] = float(np.percentile(series, 75))
            row[f"{prefix}_P90"] = float(np.percentile(series, 90))
            row[f"{prefix}_Range"] = float(series.max() - series.min())
            row[f"{prefix}_Entropy"] = _series_entropy(series)

        row["crop_class"] = label
        row["days_after_planting"] = dap
        records.append(row)

    columns = []
    for band in S2_BANDS:
        columns += [f"S2_{band}_{m}" for m in MONTHS]
    for mod in ("NDVI", "NDWI", "S1_VV", "S1_VH", "Rain"):
        columns += [f"{mod}_{m}" for m in MONTHS]
    columns += ["Terrain_Elevation", "Terrain_Slope", "Terrain_Aspect"]
    for prefix in ("VV", "VH"):
        columns += [f"{prefix}_{s}" for s in STAT_NAMES]
    columns += ["crop_class", "days_after_planting"]

    pd.DataFrame(records, columns=columns).to_csv(path, index=False)
    print(f"[synthetic] wrote {n} rows x {len(columns)} columns -> {path}")
    return path
