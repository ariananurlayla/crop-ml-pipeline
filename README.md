# Crop Analysis from Multimodal Remote Sensing - ML Pipeline

Replication + optimization pipeline for the models reported in the
**2024 Laporan Akhir IFRI** (crop phase & commodity mapping, East Java),
built for the S1 thesis *"Crop Analysis from Multimodal Remote Sensing Imagery"*.

## Quickstart

```bash
pip install -r requirements.txt

# Demo / smoke test on synthetic data (no GEE export needed):
python main.py --synthetic 1500 --epochs 8

# Real run with your GEE-exported CSV (181-column IFRI schema):
python main.py --data ../data/banyuwangi.csv --task both --epochs 150

# NASA CropHarvest open benchmark (downloads ~168 MB on first use):
python main.py --cropharvest --task classification --epochs 30

# Full Replicate-then-Innovate run (Stage 1 binary + attention + Stage 2 chain):
python main.py --data IFRI_181_Band_Dataset_Extract.csv --task both \
    --binary --attention --chain --epochs 150
```

## NASA CropHarvest data source (open benchmark)

As an alternative to the IFRI CSV, the pipeline can train on
[NASA CropHarvest](https://github.com/nasaharvest/cropharvest) (Tseng et al.,
NeurIPS Datasets & Benchmarks 2021, Zenodo record 10251170, CC-BY-SA-4.0):

```bash
python main.py --cropharvest --task classification --epochs 150 \
    --baseline lightgbm            # lightgbm | xgboost | rf
python main.py --cropharvest --task classification --binary   # Rice vs other crops
```

- **Data.** Downloaded straight from Zenodo by `cropharvest_data.py`
  (`features.tar.gz` + `labels.geojson`, ~168 MB) - deliberately *not* via the
  `cropharvest` pip package, whose pandas/geopandas pins conflict with this
  pipeline on Windows. First use extracts ~70k per-point HDF5 arrays and
  builds a cached CSV (`data/cropharvest/cropharvest_features_o{other}_s{seed}.csv`);
  expect 10-20 minutes once, seconds afterwards. `--no-download` fails cleanly
  when the files are missing. `python cropharvest_data.py --labels-only` is a
  cheap pre-flight that only downloads `labels.geojson` and prints the mapped
  class counts.
- **Labels.** Mapped onto the pipeline's 4-class schema: rice -> Rice,
  maize -> Corn, sugarcane -> Sugarcane, other labelled crops -> Other
  (sampled with `--other-samples N`, default 4000, seed-controlled). Non-crop
  and unlabelled rows are excluded; unmapped label variants are printed at
  build time (extend `CROPHARVEST_LABEL_MAP` in `config.py` if needed).
- **Features.** 18 CropHarvest bands x 12 months = 216 columns (Sentinel-1
  VV/VH, Sentinel-2 B2-B12, ERA5 temperature/precipitation, SRTM
  elevation/slope, NDVI). There is no static vector, so the CNN input is
  `[B, 18, 12]` with `n_static = 0` (supported by `CropClassifier`).
- **Split.** Spatially grouped by ~1-degree lat/lon grid cells
  (`dataset.train_val_test_split_grouped`, cells from
  `cropharvest_data.spatial_grid_groups`): no geographic cell is shared
  between train/validation/test, so held-out metrics reflect generalisation
  to unseen geographies rather than spatial autocorrelation leakage. If the
  group split cannot place every class in every split (min 5 per split by
  default) it falls back to a stratified split with a loud warning, recorded
  in `metrics_summary.json`.
- **Regression/HST is not available** for this source (no
  `days_after_planting` labels): `--task classification` only; `--chain` is
  rejected.

## Replicate-then-Innovate workflow (Colab -> Kaggle)

### Phase 1 - Data extraction (GEE / Google Colab)

1. Prepare your ground-truth points as `user_points.csv` with columns
   `longitude, latitude[, crop_class][, days_after_planting]`
   (`crop_class`: 0=Rice, 1=Corn, 2=Sugarcane, 3=Other; see
   `../user_points_template.csv`). Leave the label columns empty for
   unlabelled points.
2. In Colab: `!pip install earthengine-api`, upload `gee_feature_extraction.py`
   (TA root) and `user_points.csv`, then run it. It authenticates GEE, builds
   the 181-band 2024 stack over Banyuwangi-Jember, samples at your points and
   exports `IFRI_181_Band_Dataset_Extract.csv` to Drive (folder
   `IFRI_Reproduction_Data`), then copies it to `/content`.
3. Sanity check the CSV: 183 columns (181 features + `crop_class` +
   `days_after_planting`), one row per point. If a month's columns are all
   NaN, re-check the year/AOI (e.g. S1 scenes with no VV+VH overlap).

### Phase 2 - Model training (Kaggle GPU T4)

1. On Kaggle, create two datasets:
   - zip the **crop-ml-pipeline folder itself** -> dataset `crop-ml-pipeline`
   - upload `IFRI_181_Band_Dataset_Extract.csv` -> dataset `ifri-181-band-csv`
2. Create a notebook with the **GPU T4 x2** accelerator, upload
   `kaggle/kaggle_run.ipynb` (or paste its cells) and run it. It runs:
   - **Stage 1 (commodity segmentation):** LightGBM baseline + CNN 1D with
     temporal self-attention and focal loss, binary Rice vs Non-Rice
   - **Stage 2 (crop phase):** the HST regressor (MSE), trained only on
     pixels Stage 1 predicted as Rice
3. Download the `outputs/run_<ts>/` folder.

### Phase 3 - Baseline evaluation & output

Each run benchmarks itself against the IFRI 2024 numbers - binary accuracy
vs 89.29% (2-class stage) / 4-class vs 67.2% (Gradient Boosting), and
regression RMSE vs the 23-day manual flood-threshold method - and exports:
300-DPI figures plus `thesis_results_draft.md` / `thesis_results_draft.tex`
(compilable LaTeX chapter draft with the comparison table and a templated
discussion paragraph).

## What runs

| Part | Component | File |
|------|-----------|------|
| A | `CropDataset`: `[B, C, 12]` temporal tensor + `[B, 25]` static vector, per-channel normalisation fitted on train only | `dataset.py` |
| B.1 | Shallow tabular baselines on the raw features - LightGBM (IFRI replication target: 67.2%), XGBoost, or Random Forest (`--baseline`) | `baselines.py` |
| B.2 | CNN 1D multi-class classifier (backbone + GAP + concat static -> 4 classes) | `models.py` |
| C.1 | CNN 1D regressor predicting `days_after_planting` (HST) - replaces the manual flood-threshold method (baseline RMSE 23 days) | `models.py` |
| C.2 | Multi-class Focal Loss for the Rice/Corn/Sugarcane imbalance | `models.py` |
| D | Training loop: Adam + ReduceLROnPlateau, early stopping, best-checkpoint saving, accuracy / macro-F1 / per-class P-R-F1 / RMSE-MAE-R^2 | `training.py` |
| E | `outputs/run_<ts>/{weights,figures,reports}/`, `metrics_summary.json`, 300-DPI figures, `results_draft.md` + `results_draft.tex` | `reporting.py` |

Entry point and CLI: `main.py`.

## Expected CSV schema (181 features + 2 targets)

- 96 `S2_B2_Jan`..`S2_B8A_Dec` (8 Sentinel-2 bands x 12 months)
- 12 monthly `NDVI_*`, 12 `NDWI_*`, 12 `S1_VV_*`, 12 `S1_VH_*`, 12 `Rain_*`
- 3 `Terrain_*` static bands
- 22 annual `VV_*` / `VH_*` radar statistics & entropy bands (any exact names -
  they are discovered dynamically)
- Targets: `crop_class` (0=Rice, 1=Corn, 2=Sugarcane, 3=Other),
  `days_after_planting` (may be NaN outside rice plots)

## Design notes

- **Temporal tensor shape.** The spec's `[B, 12, 12]` covers the five listed
  modalities (8 S2 + NDVI + NDWI + VV + VH = 12 channels x 12 months = 144 of
  the 156 temporal features). By default (`--rain-mode temporal_channel`) the
  12 monthly `Rain` bands become a 13th temporal channel so **all** features
  are used and the static vector stays at exactly 25. Use `--rain-mode static`
  for the literal `[B, 12, 12]` tensor (rain then joins the static vector).
- **Split.** 80/10/10 stratified - the 20% held-out block is split into a
  validation half (early stopping) and a test half (final metrics), so the
  final test set is never seen during training.
- **Baseline comparison.** The shallow GB and the CNN use the *same* seeded
  train/test split, so the delta reported against the IFRI 67.2% baseline is
  a like-for-like comparison.
- **NaN handling.** Missing months (cloud gaps) are filled with the
  per-column median before normalisation. Regression rows with NaN
  `days_after_planting` are dropped; predictions are clamped at 0 (HST cannot
  be negative).
- **Focal loss** is used without `alpha`: the `(1-p_t)^gamma` factor alone
  down-weights the easy majority class. `--loss ce_balanced` gives
  inverse-frequency class weighting instead; `--loss ce` is plain
  cross-entropy.
- **`--binary`** (Stage 1) remaps labels to the IFRI 2-class stage
  (1=Rice, 0=Non-Rice); **`--attention`** adds multi-head temporal
  self-attention over the monthly sequence before the global average pool
  (the final MaxPool is skipped so the sequence keeps length 3 for the
  attention layer); **`--chain`** (Stage 2, needs `--task both`) trains the
  HST regressor only on rows Stage 1 predicts as Rice.
- **Figures** follow a validated colourblind-safe palette; learning curves
  use two side-by-side panels rather than a dual y-axis (loss and accuracy
  live on unrelated scales - a shared axis would invent a spurious
  relationship).

## Outputs per run

```
outputs/run_20260813_153000/
├── weights/
│   ├── best_classifier.pt     # checkpoint: state_dict + epoch + val_loss + config
│   ├── best_regressor.pt
│   └── baseline_gbdt.pkl      # fitted shallow baseline
├── figures/
│   ├── learning_curves.png            # train/val loss + accuracy panels
│   ├── confusion_matrix.png           # Rice/Corn/Sugarcane/Other heatmap
│   ├── learning_curves_regression.png
│   └── regression_analysis.png        # actual vs predicted HST + 1:1 line
├── reports/
│   ├── thesis_results_draft.md
│   └── thesis_results_draft.tex       # compilable (figures via ../figures/)
└── metrics_summary.json               # config + all metrics, machine-readable
```

The generated `thesis_results_draft.tex`/`.md` contain the methodology
outline, a table comparing this run against the IFRI baselines (67.2% GB
accuracy / 89.29% 2-class stage / 23-day manual phase-mapping RMSE, cited as
[1,2]), and a template-driven discussion paragraph including the
Corn-vs-Sugarcane confusion analysis.
