"""
End-to-end CLI for the crop-analysis pipeline (Parts A-E).

Examples:
  python main.py --data exports/banyuwangi.csv --task both --epochs 150
  python main.py --synthetic 1500 --epochs 8        # smoke test / demo

Every run creates outputs/run_<timestamp>/ containing weights/, figures/,
reports/ (Markdown + LaTeX thesis draft) and metrics_summary.json.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import dataset as ds
import synthetic
from baselines import run_tabular_baseline
from config import (CLASS_NAMES, CROPHARVEST_CHANNELS, CROPHARVEST_LICENSE,
                    CROPHARVEST_REGIONS, CROPHARVEST_ZENODO_ID,
                    IFRI_BASELINES, TARGET_REGRESSION, TrainingConfig)
from models import CropClassifier, CropRegressor
from training import (build_loss_fn, evaluate_final, predict, resolve_device,
                      set_seed, train_model)
import reporting


def parse_args():
    p = argparse.ArgumentParser(description="Crop analysis - multimodal RS ML pipeline")
    p.add_argument("--data", type=str, default=None,
                   help="path to the GEE-exported CSV (181-column IFRI schema)")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="generate N synthetic rows and train on them (demo mode)")
    p.add_argument("--cropharvest", action="store_true",
                   help="use the NASA CropHarvest dataset (Zenodo record 10251170, "
                        "CC-BY-SA-4.0) as the data source; downloads ~168 MB on first use")
    p.add_argument("--region", choices=["global", "sea"], default="global",
                   help="CropHarvest: restrict to a regional bbox from "
                        "config.CROPHARVEST_REGIONS ('sea' = Southeast Asia/ASEAN)")
    p.add_argument("--cropharvest-root", default="data/cropharvest",
                   help="directory for CropHarvest files (default: data/cropharvest)")
    p.add_argument("--other-samples", type=int, default=4000, metavar="N",
                   help="CropHarvest: number of sampled 'Other' rows (default 4000)")
    p.add_argument("--no-download", action="store_true",
                   help="CropHarvest: fail instead of downloading missing files")
    p.add_argument("--task", choices=["classification", "regression", "both"],
                   default="both")
    p.add_argument("--loss", choices=["ce", "ce_balanced", "focal"], default="focal",
                   help="classification loss")
    p.add_argument("--baseline", choices=["lightgbm", "xgboost", "rf", "extratrees", "all"],
                   default="lightgbm", help="shallow tabular baseline model "
                   "('all' fits LightGBM + XGBoost + RandomForest + ExtraTrees)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rain-mode", choices=["temporal_channel", "static"],
                   default="temporal_channel",
                   help="temporal_channel: [B,13,12] tensor; static: literal [B,12,12] "
                        "(ignored with --cropharvest)")
    p.add_argument("--outputs", default="outputs", help="output base directory")
    p.add_argument("--device", default=None, help="torch device (auto when omitted)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="skip the shallow Gradient Boosting baseline")
    p.add_argument("--binary", action="store_true",
                   help="Stage 1: binary Rice vs Non-Rice classification (IFRI 2-class stage)")
    p.add_argument("--attention", action="store_true",
                   help="add temporal self-attention over the monthly sequence (innovation)")
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument("--chain", action="store_true",
                   help="Stage 2: train the HST regressor only on Stage-1 Rice-predicted pixels")
    return p.parse_args()


def _history_json(history):
    return {k: [round(float(v), 4) for v in vals] for k, vals in history.items()}


def main():
    args = parse_args()
    if sum([bool(args.data), args.synthetic > 0, args.cropharvest]) != 1:
        sys.exit("provide exactly one data source: --data <csv>, --synthetic N, "
                 "or --cropharvest")
    if args.cropharvest and args.task != "classification":
        sys.exit("--cropharvest supports only --task classification "
                 "(no days_after_planting labels)")
    if args.cropharvest and args.chain:
        sys.exit("--chain requires --task both and HST labels - "
                 "not available for CropHarvest")

    run_dir = reporting.make_run_dir(args.outputs)
    config = TrainingConfig(
        data_path=args.data or "", output_dir=args.outputs, task=args.task,
        rain_mode=args.rain_mode, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, loss=args.loss, seed=args.seed, device=args.device,
        baseline=args.baseline,
        binary=args.binary, use_attention=args.attention, num_heads=args.heads)
    set_seed(config.seed)
    device = resolve_device(config.device)

    cropharvest_kwargs = {}
    split_groups = None
    if args.synthetic:
        csv_path = os.path.join(run_dir, "synthetic_data.csv")
        synthetic.generate_synthetic_csv(csv_path, n=args.synthetic, seed=config.seed)
        df = pd.read_csv(csv_path)
        data_summary = {"rows": int(len(df)), "source": os.path.basename(csv_path)}
    elif args.cropharvest:
        import cropharvest_data as ch
        root = ch.ensure_cropharvest(args.cropharvest_root, download=not args.no_download)
        df = ch.build_dataframe(root, other_samples=args.other_samples, seed=config.seed)
        region_info = {"region": args.region}
        if args.region == "sea":
            bbox = CROPHARVEST_REGIONS["sea"]
            df = ch.filter_region(df, bbox)
            if len(df) == 0:
                sys.exit("[main] --region sea matched 0 CropHarvest rows - "
                         "check CROPHARVEST_REGIONS in config.py")
            region_info["bbox"] = bbox
        config.data_path = os.path.join(root, "cropharvest_features.csv")
        split_groups = ch.spatial_grid_groups(df)
        cropharvest_kwargs = dict(
            channels=CROPHARVEST_CHANNELS,
            ignore_cols=ch.META_COLS + [ch.RAW_LABEL_COL])
        data_summary = {
            "rows": int(len(df)), "source": "cropharvest",
            "zenodo_id": CROPHARVEST_ZENODO_ID, "license": CROPHARVEST_LICENSE,
            "class_counts": {CLASS_NAMES[k]: int(v) for k, v in
                             df["crop_class"].value_counts().sort_index().items()},
            "other_samples": args.other_samples,
            "n_source_datasets": int(df["source_dataset"].nunique()),
            "n_spatial_cells": int(split_groups.nunique()),
            **region_info,
        }
    else:
        df = pd.read_csv(args.data)
        data_summary = {"rows": int(len(df)), "source": os.path.basename(args.data)}
    print(f"[main] loaded {len(df)} rows from {data_summary['source']}")
    print(f"[main] torch device: {device}")

    summary = {
        "run_id": os.path.basename(run_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "data": data_summary,
        "classification": {}, "regression": {},
        "figures": [], "reports": [],
    }
    extra_reports = []

    temporal_cols, static_cols, channels = ds.discover_feature_columns(
        df, include_rain=(config.rain_mode == "temporal_channel"),
        channels=cropharvest_kwargs.get("channels"),
        ignore_cols=cropharvest_kwargs.get("ignore_cols", ()))
    feature_cols = temporal_cols + static_cols

    # ------------------------------------------------------------------
    # classification: shallow GB baseline + CNN 1D (Part B + C.2)
    # ------------------------------------------------------------------
    if config.task in ("classification", "both"):
        print("\n=== [1/2] CLASSIFICATION: shallow baseline + CNN 1D ===")
        class_names = ["NonRice", "Rice"] if args.binary else list(CLASS_NAMES)
        print(f"[main] Stage 1 targets: {len(class_names)} classes {class_names}"
              + (" | temporal self-attention ON" if args.attention else ""))
        if args.cropharvest:
            train_df, val_df, test_df, split_info = ds.train_val_test_split_grouped(
                df, groups=split_groups, config=config,
                min_per_class=(1 if args.region == "sea" else None),
                singletons_to_train=(args.region == "sea"),
                row_tolerance=(0.15 if args.region == "sea" else None))
            summary["data"].update({"split": split_info["split_type"],
                                    "split_info": split_info})
        else:
            train_df, val_df, test_df = ds.train_val_test_split(df, config)
            summary["data"]["split"] = "stratified"

        baseline_res = None
        baseline_res_list = []
        if not args.skip_baseline:
            model_names = (["lightgbm", "xgboost", "rf", "extratrees"]
                           if config.baseline == "all" else [config.baseline])
            for mi, mname in enumerate(model_names):
                res, fitted = run_tabular_baseline(
                    train_df, test_df, config, feature_cols,
                    model_name=mname, binary=args.binary)
                res["model_key"] = mname
                baseline_res_list.append(res)
                if mi == 0:
                    baseline_res, baseline_model = res, fitted
                pkl_name = ("baseline_gbdt.pkl" if mname == "lightgbm"
                            else f"baseline_{mname}.pkl")
                with open(os.path.join(run_dir, "weights", pkl_name), "wb") as f:
                    pickle.dump(fitted, f)
                if len(model_names) > 1 and not args.binary:
                    fig_cm_m = os.path.join(run_dir, "figures",
                                            f"confusion_matrix_{mname}.png")
                    reporting.plot_confusion_matrix(
                        np.array(res["confusion_matrix"]), fig_cm_m, class_names)
                    summary["figures"].append(fig_cm_m)
                    cr_path = os.path.join(run_dir, "reports",
                                           f"classification_report_{mname}.json")
                    reporting.save_json(classification_report(
                        res["y_true"], res["y_pred"],
                        labels=range(len(class_names)), target_names=class_names,
                        output_dict=True, zero_division=0), cr_path)
                    extra_reports.append(cr_path)
            if len(model_names) > 1:
                comp_path = os.path.join(run_dir, "reports", "baseline_comparison.csv")
                pd.DataFrame(baseline_res_list).drop(
                    columns=["y_true", "y_pred"]).to_csv(comp_path, index=False)
                extra_reports.append(comp_path)

        train_dl, val_dl, test_dl, train_ds = ds.build_dataloaders(
            train_df, val_df, test_df, config, "classification",
            channels=cropharvest_kwargs.get("channels"),
            ignore_cols=cropharvest_kwargs.get("ignore_cols", ()))
        print(f"[main] CNN 1D input: [{train_ds.n_channels}, {train_ds.seq_len}] temporal "
              f"({', '.join(train_ds.channels)}) + [{train_ds.n_static}] static | "
              f"train {len(train_ds)} / val {len(val_dl.dataset)} / test {len(test_dl.dataset)}")

        model = CropClassifier(config, in_channels=train_ds.n_channels,
                               n_static=train_ds.n_static,
                               num_classes=len(class_names)).to(device)
        criterion, loss_name = build_loss_fn(config, "classification", train_ds.y,
                                             len(class_names))
        history, best_epoch = train_model(
            model, train_dl, val_dl, criterion, "classification", config, device,
            os.path.join(run_dir, "weights", "best_classifier.pt"))

        metrics = evaluate_final(model, test_dl, "classification", device,
                                 class_names)
        metrics.update({
            "best_epoch": best_epoch,
            "final_train_loss": round(history["train_loss"][-1], 4),
            "final_val_loss": round(history["val_loss"][-1], 4),
            "train_val_gap_pct": round(history["train_metric"][-1] - history["val_metric"][-1], 2),
        })
        if args.cropharvest:
            if baseline_res is not None:
                print(f"[main] CNN 1D test: acc {metrics['test_accuracy_pct']}% | "
                      f"macro-F1 {metrics['macro_f1_pct']}% | "
                      f"vs {baseline_res['model']} {baseline_res['test_accuracy_pct']}%: "
                      f"{metrics['test_accuracy_pct'] - baseline_res['test_accuracy_pct']:+.1f} pp")
            else:
                print(f"[main] CNN 1D test: acc {metrics['test_accuracy_pct']}% | "
                      f"macro-F1 {metrics['macro_f1_pct']}%")
        else:
            base_acc = (IFRI_BASELINES["binary_accuracy_pct"] if args.binary
                        else IFRI_BASELINES["classification_accuracy_pct"])
            base_name = "IFRI 2-class stage" if args.binary else "IFRI GB"
            print(f"[main] CNN 1D test: acc {metrics['test_accuracy_pct']}% | "
                  f"macro-F1 {metrics['macro_f1_pct']}% | "
                  f"vs {base_name} {base_acc}%: "
                  f"{metrics['test_accuracy_pct'] - base_acc:+.1f} pp")

        fig_lc = os.path.join(run_dir, "figures", "learning_curves.png")
        reporting.plot_learning_curves(history, "Accuracy (%)", fig_lc)
        fig_cm = os.path.join(run_dir, "figures", "confusion_matrix.png")
        reporting.plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                                        fig_cm, class_names)
        if args.cropharvest:
            fig_split = os.path.join(run_dir, "figures", "split_map.png")
            reporting.plot_split_map(train_df, val_df, test_df, fig_split)
            summary["figures"] += [fig_split]
            if args.region == "sea":
                fig_sea = os.path.join(run_dir, "figures",
                                       "sea_sample_distribution.png")
                reporting.plot_sea_distribution(df, fig_sea)
                sea_copy = os.path.join(args.outputs, "sea_sample_distribution.png")
                shutil.copyfile(fig_sea, sea_copy)
                summary["figures"].append(fig_sea)
                summary["sea_map_copy"] = sea_copy

        summary["classification"] = {
            "loss": loss_name, "metric": "accuracy",
            "class_names": class_names, "binary": args.binary,
            "cnn1d": metrics, "baseline": baseline_res,
            "baselines": baseline_res_list,
            "history": _history_json(history),
            "tensor": {"channels": train_ds.channels,
                       "n_channels": train_ds.n_channels,
                       "seq_len": train_ds.seq_len,
                       "n_static": train_ds.n_static},
        }
        summary["figures"] += [fig_lc, fig_cm]

    # ------------------------------------------------------------------
    # regression: CNN 1D -> days after planting (Part C.1)
    # ------------------------------------------------------------------
    if config.task in ("regression", "both"):
        print("\n=== [2/2] REGRESSION: CNN 1D -> days_after_planting (HST) ===")
        dap_df = df.dropna(subset=[TARGET_REGRESSION]).reset_index(drop=True)
        chain_info = None
        if args.chain:
            # Stage-2 gate (IFRI hierarchical design): keep only the
            # HST-labelled rows that the Stage-1 classifier predicts as Rice.
            if config.task != "both":
                sys.exit("--chain requires --task both (Stage 1 must run first)")
            gate_ds = ds.CropDataset(dap_df, config, "classification",
                                     train_ds.norm_t, train_ds.norm_s)
            gate_preds = predict(model, gate_ds, device)
            rice_idx = class_names.index("Rice")
            keep = gate_preds == rice_idx
            print(f"[main] Stage-2 gate: {int(keep.sum())}/{len(dap_df)} "
                  f"HST-labelled rows predicted as Rice")
            if keep.sum() < 40:
                sys.exit("[main] too few Rice-predicted rows for Stage 2 - "
                         "check Stage 1 accuracy or the ground-truth labels")
            dap_df = dap_df[keep].reset_index(drop=True)
            chain_info = {"hst_labelled_rows": int(len(gate_preds)),
                          "rice_predicted_rows": int(keep.sum())}
        train_df, val_df, test_df = ds.train_val_test_split(dap_df, config)
        train_dl, val_dl, test_dl, train_ds = ds.build_dataloaders(
            train_df, val_df, test_df, config, "regression")
        print(f"[main] CNN 1D input: [{train_ds.n_channels}, {train_ds.seq_len}] temporal "
              f"+ [{train_ds.n_static}] static | "
              f"train {len(train_ds)} / val {len(val_dl.dataset)} / test {len(test_dl.dataset)}")

        model = CropRegressor(config, in_channels=train_ds.n_channels,
                              n_static=train_ds.n_static).to(device)
        criterion, loss_name = build_loss_fn(config, "regression")
        history, best_epoch = train_model(
            model, train_dl, val_dl, criterion, "regression", config, device,
            os.path.join(run_dir, "weights", "best_regressor.pt"))

        metrics = evaluate_final(model, test_dl, "regression", device)
        metrics.update({
            "best_epoch": best_epoch,
            "final_train_loss": round(history["train_loss"][-1], 4),
            "final_val_loss": round(history["val_loss"][-1], 4),
            "train_val_gap": round(history["train_metric"][-1] - history["val_metric"][-1], 2),
        })
        print(f"[main] CNN 1D test: RMSE {metrics['test_rmse_days']} days | "
              f"MAE {metrics['test_mae_days']} days | R^2 {metrics['test_r2']} | "
              f"vs IFRI flood-threshold {IFRI_BASELINES['regression_rmse_days']} days: "
              f"{metrics['test_rmse_days'] - IFRI_BASELINES['regression_rmse_days']:+.1f} days")

        fig_lc = os.path.join(run_dir, "figures", "learning_curves_regression.png")
        reporting.plot_learning_curves(history, "MAE (days)", fig_lc)
        fig_reg = os.path.join(run_dir, "figures", "regression_analysis.png")
        reporting.plot_regression_analysis(metrics["y_true"], metrics["y_pred"], fig_reg)

        summary["regression"] = {
            "loss": loss_name, "metric": "mae",
            "chain": chain_info,
            "cnn1d": metrics, "history": _history_json(history),
            "tensor": {"channels": train_ds.channels,
                       "n_channels": train_ds.n_channels,
                       "seq_len": train_ds.seq_len,
                       "n_static": train_ds.n_static},
        }
        summary["figures"] += [fig_lc, fig_reg]

    # ------------------------------------------------------------------
    # Part E: JSON summary + thesis draft (Markdown + LaTeX)
    # ------------------------------------------------------------------
    print("\n=== [3/3] REPORTS ===")
    md_path, tex_path = reporting.generate_thesis_report(summary, run_dir)
    summary["reports"] = [md_path, tex_path] + extra_reports
    reporting.save_json(summary, os.path.join(run_dir, "metrics_summary.json"))

    print(f"\nDone. Everything is under: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
