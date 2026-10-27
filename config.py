"""
Central configuration for the crop-analysis deep-learning pipeline.

This pipeline replicates and extends the models reported in the
"2024 Laporan Akhir IFRI" (crop phase & commodity mapping, East Java):

  * Shallow baseline : Gradient Boosting on the raw 181-column table
                       (IFRI reported ~67.2% test accuracy).
  * CNN 1D           : [B, C, 12] temporal tensor + static features
                       -> 4 crop classes (Rice, Corn, Sugarcane, Other).
  * CNN 1D regressor : predicts 'days_after_planting' (HST) directly,
                       targeting the IFRI flood-threshold RMSE of ~23 days.

All knobs live in :class:`TrainingConfig`; the CLI (main.py) overrides a few.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

MONTHS: List[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CLASS_NAMES: List[str] = ["Rice", "Corn", "Sugarcane", "Other"]  # crop_class 0..3
NUM_CLASSES: int = len(CLASS_NAMES)

TARGET_CLASS: str = "crop_class"
TARGET_REGRESSION: str = "days_after_planting"

S2_BANDS: List[str] = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]

TERRAIN_FEATURES: List[str] = ["Terrain_Elevation", "Terrain_Slope", "Terrain_Aspect"]

# Reported numbers from the 2024 IFRI final report, used by the automated
# thesis-draft generator as the baseline row to compare against.
IFRI_BASELINES: Dict[str, float] = {
    "classification_accuracy_pct": 67.2,   # Gradient Boosting, skema 2, test accuracy
    "binary_accuracy_pct": 89.29,          # 2-class padi/non-padi stage (CNN+LSTM, high-res data)
    "regression_rmse_days": 23.0,          # flood-threshold phase mapping vs field data
}
IFRI_REFERENCE: str = "2024 Laporan Akhir IFRI"

# --- NASA CropHarvest (Tseng et al., NeurIPS D&B 2021; Zenodo record 10251170) ---
# Loaded directly from Zenodo by cropharvest_data.py - deliberately NOT via the
# ``cropharvest`` pip package, whose pandas<2.0 / geopandas / rasterio pins
# conflict with this pipeline and need a conda env on Windows.
CROPHARVEST_ZENODO_ID: str = "10251170"
CROPHARVEST_FILES: Dict[str, str] = {
    "features": "features.tar.gz",   # per-point h5 arrays, shape (12 months, 18 bands)
    "labels": "labels.geojson",      # lat/lon, source dataset, crop label per point
}
CROPHARVEST_DEFAULT_ROOT: str = "data/cropharvest"

# Exact band order of the (12, 18) arrays in features/arrays/*.h5
CROPHARVEST_CHANNELS: List[str] = [
    "VV", "VH",                                # Sentinel-1 SAR
    "B2", "B3", "B4", "B5", "B6", "B7", "B8",  # Sentinel-2 optical
    "B8A", "B9", "B11", "B12",
    "temperature_2m", "total_precipitation",   # ERA5 climate
    "elevation", "slope",                      # SRTM terrain
    "NDVI",
]

# Raw label string -> pipeline class. Rice must stay 0 so the existing
# ``y == 0`` -> Rice binary remap keeps working; every other labelled crop
# (is_crop=True) becomes 3 (Other).
CROPHARVEST_LABEL_MAP: Dict[str, int] = {
    "rice": 0, "paddy rice": 0,
    "maize": 1, "corn": 1, "maïs": 1, "mais": 1,
    "sugarcane": 2, "canne à sucre": 2, "canne a sucre": 2,
}

CROPHARVEST_REFERENCE: str = ("G. Tseng, I. Zvonkov, C. L. Nakalembe, H. Kerner, "
                              "CropHarvest: A global dataset for crop-type classification, "
                              "NeurIPS Datasets and Benchmarks, 2021.")
CROPHARVEST_LICENSE: str = "CC-BY-SA-4.0"

# Expected 181-column layout (used only for sanity warnings):
#   96 S2 spectral (8 bands x 12) + 12 NDVI + 12 NDWI + 12 VV + 12 VH + 12 Rain = 156 temporal
#   3 terrain + 22 annual radar statistics/entropy = 25 static
EXPECTED_TEMPORAL: int = 156
EXPECTED_STATIC: int = 25
EXPECTED_TOTAL: int = 181


def default_temporal_channels(include_rain: bool = True) -> List[str]:
    """Canonical ordering of the temporal channel prefixes.

    The original spec asks for a [B, 12, 12] tensor for the five modalities
    S2 (8 bands) + NDVI + NDWI + VV + VH - i.e. 144 of the 156 temporal
    features. The 12 monthly ``Rain`` (CHIRPS) bands are left over. With
    ``include_rain=True`` (default) Rain becomes a 13th temporal channel so
    that every feature is used and the static vector stays at exactly 25.
    Set ``rain_mode='static'`` in the config to get the literal [B, 12, 12]
    tensor with Rain folded into the static vector instead.
    """
    channels: List[str] = [f"S2_{b}" for b in S2_BANDS] + ["NDVI", "NDWI", "S1_VV", "S1_VH"]
    if include_rain:
        channels.append("Rain")
    return channels


@dataclass
class TrainingConfig:
    # --- data / io ---
    data_path: str = "data.csv"
    output_dir: str = "outputs"
    task: str = "both"                        # 'classification' | 'regression' | 'both'
    rain_mode: str = "temporal_channel"       # 'temporal_channel' -> [B, 13, 12]
                                              # 'static'            -> [B, 12, 12], rain joins static vector
    # --- data split (80% train; the 20% held-out is 10% val + 10% test) ---
    seed: int = 42
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    # --- architecture ---
    conv_channels: List[int] = None           # set below in __post_init__
    kernel_size: int = 5
    dropout: float = 0.2
    fc_hidden: int = 128
    # --- optimization ---
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    early_stop_patience: int = 20
    lr_patience: int = 10
    lr_factor: float = 0.5
    loss: str = "focal"                       # 'ce' | 'ce_balanced' | 'focal'
    focal_gamma: float = 2.0
    baseline: str = "lightgbm"                # shallow baseline: 'lightgbm' | 'xgboost' | 'rf'
    min_class_coverage: int = 5               # grouped split: min per-class count per split
    binary: bool = False                      # Stage 1: Rice vs Non-Rice (2 classes)
    use_attention: bool = False               # temporal self-attention over the monthly sequence
    num_heads: int = 4                        # attention heads (must divide the last conv width)
    num_workers: int = 0
    device: Optional[str] = None              # auto-detect when None

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]

    def to_dict(self) -> dict:
        return asdict(self)
