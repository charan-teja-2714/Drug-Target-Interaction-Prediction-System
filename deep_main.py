#!/usr/bin/env python3
"""
deep_main.py

Deep Learning pipeline for Drug-Target Interaction prediction.
Runs INDEPENDENTLY from main.py (existing ML pipeline is unchanged).

Pipeline:
  1. Load dataset
  2. Extract ChemBERTa + ESM2 embeddings  (cached after first run)
  3. Train DTI Fusion Network (PyTorch, GPU)
  4. Also train LightGBM on the same embeddings for comparison
  5. Save results to results/metrics_deep.csv

Usage:
    python deep_main.py

Requirements (install separately):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier

from src.data_loader import DataLoader as DTIDataLoader
from src.embeddings import get_embeddings
from src.deep_model import DTIFusionNet
from src.config import RANDOM_STATE, TEST_SIZE, RESULTS_DIR, SAVED_MODELS_DIR, PLOTS_DIR
from src.utils import setup_logging

import joblib

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BATCH_SIZE      = 128
EPOCHS          = 100
LEARNING_RATE   = 2e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 15        # early stopping patience
EMBED_CACHE_DIR = __import__("pathlib").Path("data/processed/embeddings")


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class DTIEmbeddingDataset(Dataset):
    def __init__(self, drug_embs, protein_embs, labels=None):
        self.drug    = torch.tensor(drug_embs,    dtype=torch.float32)
        self.protein = torch.tensor(protein_embs, dtype=torch.float32)
        self.labels  = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.drug)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.drug[idx], self.protein[idx], self.labels[idx]
        return self.drug[idx], self.protein[idx]


# ------------------------------------------------------------------
# Early Stopping
# ------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int, model_path):
        self.patience    = patience
        self.model_path  = model_path
        self.best_loss   = float("inf")
        self.counter     = 0
        self.best_epoch  = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - 1e-4:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_path)
            return False    # continue
        else:
            self.counter += 1
            return self.counter >= self.patience   # True = stop


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0

    for drug, protein, labels in loader:
        drug, protein, labels = drug.to(device), protein.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(drug, protein)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * len(labels)
        total_correct += ((logits.sigmoid() >= 0.5) == labels.bool()).sum().item()
        n             += len(labels)

    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []

    for drug, protein, labels in loader:
        drug, protein, labels = drug.to(device), protein.to(device), labels.to(device)
        logits = model(drug, protein)
        loss   = criterion(logits, labels)
        probs  = logits.sigmoid()

        total_loss  += loss.item() * len(labels)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend((probs >= 0.5).cpu().numpy().astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))

    n      = len(all_labels)
    acc    = accuracy_score(all_labels, all_preds)
    auc    = roc_auc_score(all_labels, all_probs)
    f1     = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / n, acc, auc, f1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load data ----
    logger.info("Loading dataset...")
    loader = DTIDataLoader()
    df     = loader.load_data()
    y      = df["interaction"].values.astype(int)

    # ---- Extract / load embeddings ----
    logger.info("Getting pre-trained embeddings  (first run downloads models ~500MB)...")
    drug_embs, protein_embs = get_embeddings(df, device, EMBED_CACHE_DIR)

    drug_dim    = drug_embs.shape[1]
    protein_dim = protein_embs.shape[1]
    logger.info(f"Drug dim={drug_dim}, Protein dim={protein_dim}")

    # ---- Train / val / test split ----
    # 80% train (of which 10% is used as validation), 20% test
    idx    = np.arange(len(y))
    idx_tr, idx_test, y_tr, y_test = train_test_split(
        idx, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tr, y_tr, test_size=0.1, random_state=RANDOM_STATE, stratify=y_tr
    )

    X_train_d, X_val_d, X_test_d = drug_embs[idx_train],    drug_embs[idx_val],    drug_embs[idx_test]
    X_train_p, X_val_p, X_test_p = protein_embs[idx_train], protein_embs[idx_val], protein_embs[idx_test]

    logger.info(
        f"Split — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}"
    )

    # ---- DataLoaders ----
    train_ds = DTIEmbeddingDataset(X_train_d, X_train_p, y_train)
    val_ds   = DTIEmbeddingDataset(X_val_d,   X_val_p,   y_val)
    test_ds  = DTIEmbeddingDataset(X_test_d,  X_test_p,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # ====================================================================
    # 1.  TRAIN DTI FUSION NETWORK
    # ====================================================================
    logger.info("\n" + "="*60)
    logger.info("Training DTI Fusion Network (PyTorch)")
    logger.info("="*60)

    model     = DTIFusionNet(drug_dim=drug_dim, protein_dim=protein_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01)

    model_path  = SAVED_MODELS_DIR / "dti_fusion_net.pt"
    stopper     = EarlyStopping(patience=PATIENCE, model_path=model_path)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Val Acc':>8}  {'Val AUC':>8}  {'LR':>10}")
    print("-" * 65)

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_auc, vl_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {vl_loss:>10.4f}  "
              f"{vl_acc:>8.4f}  {vl_auc:>8.4f}  {lr_now:>10.2e}")

        if stopper.step(vl_loss, model, epoch):
            print(f"\n  Early stopping at epoch {epoch}  "
                  f"(best epoch: {stopper.best_epoch})")
            break

    elapsed = time.time() - t_start
    logger.info(f"Training finished in {elapsed/60:.1f} min")

    # Load best checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))

    # ---- Test evaluation ----
    _, test_acc, test_auc, test_f1 = evaluate(model, test_loader, criterion, device)

    # Full metrics on test set
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for drug, protein, _ in test_loader:
            logits = model(drug.to(device), protein.to(device))
            probs  = logits.sigmoid().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))

    nn_results = {
        "model":     "DTI_FusionNet",
        "accuracy":  accuracy_score(y_test, all_preds),
        "precision": precision_score(y_test, all_preds, zero_division=0),
        "recall":    recall_score(y_test, all_preds, zero_division=0),
        "f1_score":  f1_score(y_test, all_preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, all_probs),
    }

    print(f"\n{'='*60}")
    print(f"  DTI FUSION NET — TEST RESULTS ({len(y_test):,} samples)")
    print(f"{'='*60}")
    for k, v in nn_results.items():
        if k != "model":
            print(f"  {k:<12}: {v:.4f}")
    print(f"{'='*60}\n")

    # ====================================================================
    # 2.  LGBM ON EMBEDDINGS  (baseline comparison)
    # ====================================================================
    logger.info("Training LightGBM on embeddings for comparison...")

    X_tr_emb  = np.hstack([X_train_d, X_train_p])
    X_val_emb = np.hstack([X_val_d,   X_val_p])
    X_te_emb  = np.hstack([X_test_d,  X_test_p])

    lgbm = LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        num_leaves=127, feature_fraction=0.6,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, verbose=-1, random_state=RANDOM_STATE,
    )
    lgbm.fit(
        np.vstack([X_tr_emb, X_val_emb]),
        np.concatenate([y_train, y_val]),
    )
    lgbm_pred  = lgbm.predict(X_te_emb)
    lgbm_probs = lgbm.predict_proba(X_te_emb)[:, 1]

    lgbm_results = {
        "model":     "LightGBM_on_Embeddings",
        "accuracy":  accuracy_score(y_test, lgbm_pred),
        "precision": precision_score(y_test, lgbm_pred, zero_division=0),
        "recall":    recall_score(y_test, lgbm_pred, zero_division=0),
        "f1_score":  f1_score(y_test, lgbm_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, lgbm_probs),
    }

    print(f"\n{'='*60}")
    print(f"  LIGHTGBM ON EMBEDDINGS — TEST RESULTS")
    print(f"{'='*60}")
    for k, v in lgbm_results.items():
        if k != "model":
            print(f"  {k:<12}: {v:.4f}")
    print(f"{'='*60}\n")

    # ====================================================================
    # 3.  SAVE RESULTS
    # ====================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([nn_results, lgbm_results])
    results_df.to_csv(RESULTS_DIR / "metrics_deep.csv", index=False)
    logger.info(f"Saved results -> results/metrics_deep.csv")

    joblib.dump(lgbm, SAVED_MODELS_DIR / "lgbm_on_embeddings.joblib")

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].axvline(stopper.best_epoch - 1, color="gray", linestyle="--",
                    label=f"Best epoch ({stopper.best_epoch})")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_acc"],  label="Val Accuracy")
    axes[1].plot(history["val_auc"],  label="Val ROC-AUC")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deep_training_curves.png", dpi=150)
    plt.close()
    logger.info("Saved training curves -> results/plots/deep_training_curves.png")

    print("\nFinal comparison:")
    print(results_df[["model", "accuracy", "f1_score", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
