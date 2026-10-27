"""
Part B.1 - Shallow tabular baselines.

Three selectable models (``--baseline``): LightGBM (the IFRI replication
target ~67.2% test accuracy, skema 2), XGBoost, and Random Forest. All
consume the raw feature table directly - LightGBM/XGBoost handle NaN
natively; Random Forest gets per-column median imputation fitted on train
only. LightGBM falls back to scikit-learn's HistGradientBoostingClassifier
(both histogram-based GBDTs).
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import CLASS_NAMES, NUM_CLASSES, TARGET_CLASS


def _gbdt_model(seed: int):
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbosity=-1, n_jobs=-1)
        return model, "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=seed)
        return model, "HistGradientBoosting"


def _xgb_model(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("[baseline] xgboost is required for --baseline xgboost - "
                 "run: pip install xgboost")
    model = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        n_jobs=-1, random_state=seed, tree_method="hist")
    return model, "XGBoost"


def _rf_model(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
    return model, "RandomForest"


def run_tabular_baseline(train_df, test_df, config, feature_cols,
                         model_name: str = "lightgbm", binary: bool = False):
    """Fit a shallow tabular baseline on the same train/test split as the CNN.

    ``model_name``: 'lightgbm' (default, HistGB fallback), 'xgboost', or
    'rf'. With ``binary=True`` the labels are remapped to the IFRI 2-class
    stage: 1 = Rice, 0 = Non-Rice. Returns ``(results_dict, fitted_model)``.
    """
    if model_name == "lightgbm":
        model, name = _gbdt_model(config.seed)
    elif model_name == "xgboost":
        model, name = _xgb_model(config.seed)
    elif model_name == "rf":
        model, name = _rf_model(config.seed)
    else:
        raise ValueError(f"unknown baseline model '{model_name}'")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    if model_name == "rf":   # Random Forest cannot ingest NaN
        X_train = X_train.to_numpy(dtype=np.float32)
        X_test = X_test.to_numpy(dtype=np.float32)
        med = np.nanmedian(X_train, axis=0, keepdims=True)
        X_train = np.where(np.isnan(X_train), med, X_train)
        X_test = np.where(np.isnan(X_test), med, X_test)
    y_train = train_df[TARGET_CLASS].to_numpy()
    y_test = test_df[TARGET_CLASS].to_numpy()
    if binary:
        y_train = (y_train == 0).astype(np.int64)   # 1 = Rice, 0 = Non-Rice
        y_test = (y_test == 0).astype(np.int64)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    n_classes = 2 if binary else NUM_CLASSES

    results = {
        "model": name,
        "test_accuracy_pct": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "macro_f1_pct": round(float(f1_score(y_test, y_pred, average="macro",
                                             zero_division=0)) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred,
                                             labels=range(n_classes)).tolist(),
    }
    print(f"[baseline] {name} (raw {len(feature_cols)} features, {n_classes} classes): "
          f"test acc {results['test_accuracy_pct']}% | macro-F1 {results['macro_f1_pct']}%")
    return results, model


def run_gradient_boosting_baseline(train_df, test_df, config, feature_cols,
                                   binary: bool = False):
    """Backward-compatible wrapper: LightGBM baseline (IFRI replication)."""
    return run_tabular_baseline(train_df, test_df, config, feature_cols,
                                model_name="lightgbm", binary=binary)
