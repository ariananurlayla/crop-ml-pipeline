"""
Part D - training, validation and final-evaluation loops shared by the
classification and regression heads.

Every run tracks:
  * classification : loss, accuracy (%), macro-F1, per-class P/R/F1, confusion matrix
  * regression     : MSE loss, MAE / RMSE (days), R^2
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

from config import CLASS_NAMES, NUM_CLASSES
from models import FocalLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref=None) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_balanced_class_weights(labels: np.ndarray,
                                   num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights, mean-normalised to 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    return torch.as_tensor(w / w.mean(), dtype=torch.float32)


def build_loss_fn(config, task: str, train_labels: np.ndarray = None,
                  num_classes: int = NUM_CLASSES):
    """Return (criterion, loss_name) for the requested task/loss combo.

    Focal loss is used *without* alpha: the (1-p_t)^gamma factor alone already
    down-weights the easy majority class; alpha-balanced weighting is reserved
    for the explicit 'ce_balanced' option.
    """
    if task == "regression":
        return nn.MSELoss(), "mse"
    if config.loss == "focal":
        return FocalLoss(gamma=config.focal_gamma), "focal"
    if config.loss == "ce_balanced":
        w = (compute_balanced_class_weights(train_labels, num_classes)
             if train_labels is not None else None)
        return nn.CrossEntropyLoss(weight=w), "ce_balanced"
    return nn.CrossEntropyLoss(), "ce"


def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    total_loss, n = 0.0, 0
    correct, mae_sum = 0, 0.0
    for x_t, x_s, y in loader:
        x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x_t, x_s)
        if task == "classification":
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()
        else:
            pred = out.squeeze(1)
            loss = criterion(pred, y)
            mae_sum += (pred.detach() - y).abs().sum().item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    metric = (100.0 * correct / n) if task == "classification" else (mae_sum / n)
    return total_loss / n, metric


@torch.no_grad()
def evaluate(model, loader, criterion, device, task):
    model.eval()
    total_loss, n = 0.0, 0
    correct, mae_sum = 0, 0.0
    for x_t, x_s, y in loader:
        x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
        out = model(x_t, x_s)
        if task == "classification":
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()
        else:
            pred = out.squeeze(1)
            loss = criterion(pred, y)
            mae_sum += (pred.detach() - y).abs().sum().item()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    metric = (100.0 * correct / n) if task == "classification" else (mae_sum / n)
    return total_loss / n, metric


def train_model(model, train_dl, val_dl, criterion, task, config, device, ckpt_path):
    """Adam + ReduceLROnPlateau + early stopping; keeps the best state by val loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience)

    metric_name = "acc" if task == "classification" else "mae"
    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}
    best_val, best_state, best_epoch = float("inf"), None, -1
    patience_left = config.early_stop_patience
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_met = train_one_epoch(model, train_dl, criterion, optimizer, device, task)
        va_loss, va_met = evaluate(model, val_dl, criterion, device, task)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_metric"].append(tr_met)
        history["val_metric"].append(va_met)

        improved = va_loss < best_val - 1e-4
        marker = ""
        if improved:
            best_val, best_epoch = va_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.early_stop_patience
            marker = "  *best"
        else:
            patience_left -= 1

        print(f"[{task}] epoch {epoch:>3}/{config.epochs} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
              f"train_{metric_name} {tr_met:7.2f} | val_{metric_name} {va_met:7.2f}{marker}")

        if patience_left <= 0:
            print(f"[{task}] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    assert best_state is not None, "no checkpoint produced - increase --epochs"
    model.load_state_dict(best_state)
    torch.save({"epoch": best_epoch, "val_loss": best_val,
                "state_dict": best_state, "config": config.to_dict()}, ckpt_path)
    print(f"[{task}] best checkpoint (epoch {best_epoch}, val_loss {best_val:.4f}) "
          f"-> {ckpt_path} ({time.time() - t0:.1f}s)")
    return history, best_epoch


@torch.no_grad()
def evaluate_final(model, test_dl, task, device, class_names=CLASS_NAMES):
    """Final metrics on the held-out test split."""
    model.eval()
    preds, targets = [], []
    for x_t, x_s, y in test_dl:
        out = model(x_t.to(device), x_s.to(device))
        if task == "classification":
            preds.append(out.argmax(1).cpu())
        else:
            # days after planting cannot be negative
            preds.append(out.squeeze(1).clamp(min=0.0).cpu())
        targets.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    if task == "classification":
        n_classes = len(class_names)
        report = classification_report(y_true, y_pred, labels=range(n_classes),
                                       target_names=list(class_names), output_dict=True,
                                       zero_division=0)
        return {
            "test_accuracy_pct": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
            "macro_f1_pct": round(float(f1_score(y_true, y_pred, average="macro",
                                                 zero_division=0)) * 100, 2),
            "confusion_matrix": confusion_matrix(y_true, y_pred,
                                                 labels=range(n_classes)).tolist(),
            "per_class": {k: {"precision": round(v["precision"], 3),
                              "recall": round(v["recall"], 3),
                              "f1": round(v["f1-score"], 3)}
                          for k, v in report.items() if k in class_names},
        }
    return {
        "test_rmse_days": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "test_mae_days": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "test_r2": round(float(r2_score(y_true, y_pred)), 3),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


@torch.no_grad()
def predict(model, dataset, device, batch_size: int = 256) -> np.ndarray:
    """Class predictions for a whole dataset (used by the Stage-2 rice gate)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for x_t, x_s, _ in loader:
        out = model(x_t.to(device), x_s.to(device))
        preds.append(out.argmax(1).cpu())
    return torch.cat(preds).numpy()
