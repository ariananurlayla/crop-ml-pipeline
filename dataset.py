"""
Part A - PyTorch dataset for the 181-column IFRI crop table.

Feature layout (per the 2024 Laporan Akhir IFRI):

  temporal (12 monthly values per channel prefix):
    S2_B2..S2_B8A  ->  96   Sentinel-2 spectral
    NDVI, NDWI     ->  24   vegetation / water indices
    S1_VV, S1_VH   ->  24   Sentinel-1 SAR polarisations
    Rain           ->  12   CHIRPS precipitation
    ---------------------------
                     156
  static:
    Terrain_Elevation / _Slope / _Aspect ->  3
    VV_*/VH_* annual statistics & entropy -> 22
    ------------------------------------------
                                             25
  total = 181

Each sample is returned as
    x_temporal : [C, 12] float tensor   (C = 13 with Rain as a channel, else 12)
    x_static   : [25]    float tensor
    y          : class index (classification) or days_after_planting (regression)

Column discovery is dynamic: temporal columns are found as ``{channel}_{month}``
and *everything* else (apart from the targets) is treated as static, so the
exact names of the 22 radar-statistics columns do not need to be hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import (CLASS_NAMES, EXPECTED_TOTAL, MONTHS, TARGET_CLASS,
                    TARGET_REGRESSION, default_temporal_channels, TrainingConfig)


def discover_feature_columns(df: pd.DataFrame, include_rain: bool = True,
                             channels=None, ignore_cols=()):
    """Split the table into temporal columns and static columns.

    With ``channels=None`` the IFRI channel prefixes are used and the
    181-feature sanity warning applies. With an explicit ``channels`` list
    (CropHarvest: the 18 band names) those are matched instead and
    ``ignore_cols`` (meta columns such as lat/lon) never leak into static.
    Returns ``(temporal_cols, static_cols, channel_names)``.
    """
    temporal, found_channels = [], []
    prefixes = (default_temporal_channels(include_rain) if channels is None
                else list(channels))
    for prefix in prefixes:
        found = [f"{prefix}_{m}" for m in MONTHS if f"{prefix}_{m}" in df.columns]
        if found:
            temporal.extend(found)
            found_channels.append(prefix)
        else:
            print(f"[dataset] warning: no monthly columns found for channel '{prefix}'")
    static = [c for c in df.columns
              if c not in (TARGET_CLASS, TARGET_REGRESSION)
              and c not in temporal
              and c not in ignore_cols]
    if channels is None:
        if len(temporal) + len(static) != EXPECTED_TOTAL:
            print(f"[dataset] warning: found {len(temporal)} temporal + {len(static)} static "
                  f"= {len(temporal) + len(static)} features (expected {EXPECTED_TOTAL})")
    elif len(temporal) != len(channels) * len(MONTHS):
        print(f"[dataset] warning: found {len(temporal)} temporal columns "
              f"(expected {len(channels) * len(MONTHS)} = "
              f"{len(channels)} channels x {len(MONTHS)} months)")
    return temporal, static, found_channels


def _fill_median(x: np.ndarray) -> np.ndarray:
    """Replace NaN (cloud gaps / missing months) with the per-column median."""
    if x.shape[1] == 0:
        return x                     # empty static vector (e.g. CropHarvest)
    med = np.nanmedian(x, axis=0, keepdims=True)
    return np.where(np.isnan(x), med, x)


class ChannelNormalizer:
    """Z-scores columns using statistics fitted on the training split only."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "ChannelNormalizer":
        if x.shape[1] == 0:          # empty static vector (e.g. CropHarvest)
            self.mean, self.std = None, None
            return self
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x
        return (x - self.mean) / self.std


class CropDataset(Dataset):
    """PyTorch Dataset over the tabular GEE export."""

    def __init__(self, df: pd.DataFrame, config: TrainingConfig,
                 task: str = "classification",
                 norm_t: ChannelNormalizer = None, norm_s: ChannelNormalizer = None,
                 channels=None, ignore_cols=()):
        assert task in ("classification", "regression"), f"unknown task {task}"
        self.task = task
        df = df.copy()
        if task == "classification":
            df = df.dropna(subset=[TARGET_CLASS]).reset_index(drop=True)
        else:
            df = df.dropna(subset=[TARGET_REGRESSION]).reset_index(drop=True)
        assert len(df) > 0, f"[dataset] no rows with a valid {task} target"

        self.temporal_cols, self.static_cols, self.channels = discover_feature_columns(
            df, include_rain=(config.rain_mode == "temporal_channel"),
            channels=channels, ignore_cols=ignore_cols)
        self.n_channels = len(self.channels)
        self.n_static = len(self.static_cols)
        self.seq_len = len(MONTHS)

        x_t = _fill_median(df[self.temporal_cols].to_numpy(dtype=np.float32))
        x_s = _fill_median(df[self.static_cols].to_numpy(dtype=np.float32))
        self.x_t = x_t.reshape(len(df), self.n_channels, self.seq_len)
        self.x_s = x_s

        if norm_t is not None:
            self.x_t = norm_t.transform(self.x_t)
        if norm_s is not None:
            self.x_s = norm_s.transform(self.x_s)

        if task == "classification":
            y = df[TARGET_CLASS].to_numpy(dtype=np.int64)
            if config.binary:
                y = (y == 0).astype(np.int64)   # 1 = Rice, 0 = Non-Rice
            self.y = y
        else:
            self.y = df[TARGET_REGRESSION].to_numpy(dtype=np.float32)
        # expose the fitted normalisers so the Stage-2 gate can reuse them
        # when applying the Stage-1 classifier to the HST-labelled rows
        self.norm_t, self.norm_s = norm_t, norm_s

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x_t[idx]),
                torch.from_numpy(self.x_s[idx]),
                torch.tensor(self.y[idx]))


def train_val_test_split(df: pd.DataFrame, config: TrainingConfig, stratify: bool = True):
    """Stratified 80 / 10 / 10 split.

    Keeps the 80/20 train/hold-out ratio of the spec: the 20% hold-out block
    is itself split into a validation half (early stopping) and a test half
    (final metrics), so held-out data never influences training.
    Falls back to a plain random split if a class is too rare to stratify.
    """
    strat_col = TARGET_CLASS if (stratify and TARGET_CLASS in df.columns) else None
    try:
        train, rest = train_test_split(
            df, test_size=config.val_size + config.test_size,
            stratify=df[strat_col] if strat_col else None, random_state=config.seed)
        val, test = train_test_split(
            rest, test_size=config.test_size / (config.val_size + config.test_size),
            stratify=rest[strat_col] if strat_col else None, random_state=config.seed)
    except ValueError:
        print("[dataset] warning: stratified split failed (rare class) - using random split")
        train, rest = train_test_split(
            df, test_size=config.val_size + config.test_size, random_state=config.seed)
        val, test = train_test_split(
            rest, test_size=config.test_size / (config.val_size + config.test_size),
            random_state=config.seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_val_test_split_grouped(df: pd.DataFrame, groups, config: TrainingConfig,
                                 min_per_class: int = None, max_attempts: int = 25,
                                 singletons_to_train: bool = False,
                                 row_tolerance: float = None):
    """Spatially-grouped 80/10/10 split: no group (e.g. source region) is
    shared between train/val/test, so held-out metrics reflect generalisation
    to unseen geographies instead of spatial autocorrelation leakage.

    ``groups`` is an array-like (or a column name) of group ids. GroupShuffleSplit
    cannot stratify, so the split is retried with fresh seeds until every class
    present in the data has >= ``min_per_class`` samples in all three splits;
    if no attempt passes, falls back to the stratified split with a loud
    warning.

    With ``singletons_to_train=True`` (small regional subsets), a class with a
    single sample cannot populate all three splits, so those rows are pinned to
    train and the coverage check runs on the remaining classes only.

    With ``row_tolerance`` (small datasets), an attempt is also rejected when
    any split deviates from the configured row fractions by more than this
    value; if no attempt meets it, the closest coverage-passing attempt is
    used. Returns ``(train, val, test, info)``.
    """
    min_per_class = (config.min_class_coverage if min_per_class is None
                     else min_per_class)
    if isinstance(groups, str):
        groups = df[groups].to_numpy()
    y = df[TARGET_CLASS].to_numpy()
    classes_present = np.unique(y)
    counts = np.bincount(y, minlength=int(classes_present.max()) + 1)

    pool_idx = np.arange(len(df))
    forced_idx = np.array([], dtype=int)
    cov_classes = list(classes_present)
    singleton_info = None
    if singletons_to_train:
        singleton = [c for c in classes_present if counts[c] == 1]
        if singleton:
            forced_idx = np.where(np.isin(y, singleton))[0]
            pool_idx = np.setdiff1d(pool_idx, forced_idx)
            cov_classes = [c for c in classes_present if c not in singleton]
            singleton_info = {CLASS_NAMES[c]: int(counts[c]) for c in singleton}
            print(f"[dataset] singleton classes {singleton_info} "
                  f"({len(forced_idx)} rows) pinned to train - cannot cover all splits")

    def _counts(idx):
        cnt = np.bincount(y[idx], minlength=int(classes_present.max()) + 1)
        return {CLASS_NAMES[c]: int(cnt[c]) for c in classes_present}

    def _coverage_ok(train_idx, val_idx, test_idx):
        for idx in (train_idx, val_idx, test_idx):
            cnt = np.bincount(y[idx], minlength=int(classes_present.max()) + 1)
            if any(cnt[c] < min_per_class for c in cov_classes):
                return False
        return True

    info = {"split_type": "spatial-group"}
    if singleton_info:
        info["singleton_classes_to_train"] = singleton_info
    best_candidate = None
    n = len(df)
    for attempt in range(max_attempts):
        try:
            gss = GroupShuffleSplit(n_splits=1,
                                    test_size=config.val_size + config.test_size,
                                    random_state=config.seed + attempt)
            train_rel, rest_rel = next(gss.split(pool_idx, groups=groups[pool_idx]))
            rest_frac = config.test_size / (config.val_size + config.test_size)
            val_rel, test_rel = next(GroupShuffleSplit(
                n_splits=1, test_size=rest_frac,
                random_state=config.seed + attempt).split(
                    pool_idx[rest_rel], groups=groups[pool_idx[rest_rel]]))
            train_pool = pool_idx[train_rel]
            rest_pool = pool_idx[rest_rel]
            val_idx = rest_pool[val_rel]
            test_idx = rest_pool[test_rel]
            train_idx = np.concatenate([forced_idx, train_pool])
        except ValueError:
            continue
        if _coverage_ok(train_idx, val_idx, test_idx):
            dev = max(abs(len(train_idx) / n - config.train_size),
                      abs(len(val_idx) / n - config.val_size),
                      abs(len(test_idx) / n - config.test_size))
            if row_tolerance is not None:
                if best_candidate is None or dev < best_candidate[0]:
                    best_candidate = (dev, train_idx, val_idx, test_idx, attempt + 1)
                if dev > row_tolerance:
                    continue
            info.update({
                "attempts": attempt + 1,
                "train_groups": int(len(np.unique(groups[train_idx]))),
                "val_groups": int(len(np.unique(groups[val_idx]))),
                "test_groups": int(len(np.unique(groups[test_idx]))),
                "class_counts": {"train": _counts(train_idx),
                                 "val": _counts(val_idx),
                                 "test": _counts(test_idx)},
            })
            print(f"[dataset] spatial-group split OK (attempt {attempt + 1}): "
                  f"train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} rows | "
                  f"{info['train_groups']}/{info['val_groups']}/{info['test_groups']} groups")
            return (df.iloc[train_idx].reset_index(drop=True),
                    df.iloc[val_idx].reset_index(drop=True),
                    df.iloc[test_idx].reset_index(drop=True),
                    info)
    if best_candidate is not None:
        dev, train_idx, val_idx, test_idx, attempt = best_candidate
        info.update({
            "attempts": attempt,
            "train_groups": int(len(np.unique(groups[train_idx]))),
            "val_groups": int(len(np.unique(groups[val_idx]))),
            "test_groups": int(len(np.unique(groups[test_idx]))),
            "class_counts": {"train": _counts(train_idx),
                             "val": _counts(val_idx),
                             "test": _counts(test_idx)},
            "row_fraction_deviation": round(dev, 3),
        })
        print(f"[dataset] spatial-group split OK (attempt {attempt}, best of "
              f"{max_attempts}): train {len(train_idx)} / val {len(val_idx)} / "
              f"test {len(test_idx)} rows (row-fraction deviation {dev:.3f} > "
              f"tolerance {row_tolerance})")
        return (df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True),
                info)
    print(f"[dataset] WARNING: spatial-group split failed the class-coverage check "
          f"after {max_attempts} attempts - falling back to stratified split "
          "(spatial leakage possible)")
    train, val, test = train_val_test_split(df, config)
    info = {"split_type": "stratified-fallback", "attempts": max_attempts}
    return train, val, test, info


def build_dataloaders(train_df, val_df, test_df, config: TrainingConfig, task: str,
                      channels=None, ignore_cols=()):
    """Build train/val/test loaders with channel normalisation fitted on train only."""
    raw = CropDataset(train_df, config, task, channels=channels, ignore_cols=ignore_cols)
    norm_t = ChannelNormalizer().fit(raw.x_t)
    norm_s = ChannelNormalizer().fit(raw.x_s)
    train_ds = CropDataset(train_df, config, task, norm_t, norm_s, channels, ignore_cols)
    val_ds = CropDataset(val_df, config, task, norm_t, norm_s, channels, ignore_cols)
    test_ds = CropDataset(test_df, config, task, norm_t, norm_s, channels, ignore_cols)
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        assert len(ds) > 0, f"[dataset] empty {name} split for task '{task}'"
    common = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  pin_memory=False)
    train_dl = DataLoader(train_ds, shuffle=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    test_dl = DataLoader(test_ds, shuffle=False, **common)
    return train_dl, val_dl, test_dl, train_ds
