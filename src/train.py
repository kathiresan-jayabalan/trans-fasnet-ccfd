"""Train TransFASNet on the Kaggle Credit Card Fraud dataset

Random stratified 80/10/10 split, BorderlineSMOTE on the training partition, 
pretraining windows built from the full scaled dataset, 10 pretraining epochs,
10 fine-tuning epochs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from architecture import TransFASNet, nt_xent_loss_simple
from data import (
    load_and_prepare_data,
    sliding_windows,
    PretrainDataset,
    FinetuneDataset,
)

BATCH_SIZE_PRETRAIN = 512
BATCH_SIZE_FINETUNE = 256
SEQ_LEN = 8
EMBED_DIM = 128
TRANSFORMER_LAYERS = 3
NUM_HEADS = 4
MLP_DIM = 256
PROJ_DIM = 64
PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 10
LR_PRETRAIN = 3e-4
LR_FINETUNE = 1e-4
TEMP = 0.1
ALPHA_CONTRAST = 1.0
ALPHA_TEMPORAL = 1.0
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def evaluate_classifier(model, loader, device):
    model.eval()
    preds_all, probs_all, trues_all = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.float().to(device)
            yb = yb.long().to(device)

            logits = model.forward_classify_from_flat(Xb)
            prob = torch.softmax(logits, dim=1)
            out = torch.argmax(prob, dim=1)

            preds_all.append(out.cpu().numpy())
            probs_all.append(prob.cpu().numpy())
            trues_all.append(yb.cpu().numpy())

    preds = np.concatenate(preds_all)
    probs = np.concatenate(probs_all)
    trues = np.concatenate(trues_all)

    acc = (preds == trues).mean()
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    try:
        roc = roc_auc_score(trues, probs[:, 1])
    except Exception:
        roc = 0.0
    return acc, prec, rec, f1, roc


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (X_train_over, y_train_over, X_valid, y_valid,
     X_test, y_test, X_scaled, input_dim) = load_and_prepare_data(args.data_path, seed=SEED)

    print(f"Train (oversampled): {X_train_over.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    all_windows = sliding_windows(X_scaled, SEQ_LEN)
    print(f"Pretraining windows: {all_windows.shape}")

    pretrain_loader = DataLoader(
        PretrainDataset(all_windows), batch_size=BATCH_SIZE_PRETRAIN,
        shuffle=True, drop_last=True, num_workers=0,
    )
    finetune_train_loader = DataLoader(
        FinetuneDataset(X_train_over, y_train_over), batch_size=BATCH_SIZE_FINETUNE,
        shuffle=True, num_workers=0,
    )
    finetune_valid_loader = DataLoader(
        FinetuneDataset(X_valid, y_valid), batch_size=BATCH_SIZE_FINETUNE,
        shuffle=False, num_workers=0,
    )
    finetune_test_loader = DataLoader(
        FinetuneDataset(X_test, y_test), batch_size=BATCH_SIZE_FINETUNE,
        shuffle=False, num_workers=0,
    )

    model = TransFASNet(
        input_dim=input_dim, seq_len=SEQ_LEN, embed_dim=EMBED_DIM,
        n_layers=TRANSFORMER_LAYERS, n_heads=NUM_HEADS,
        mlp_dim=MLP_DIM, proj_dim=PROJ_DIM,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransFASNet (v1) trainable parameters: {total_params:,}")

    optimizer_pre = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN)
    model.train()
    print("=== Pretraining ===")
    for epoch in range(PRETRAIN_EPOCHS):
        epoch_loss = 0.0
        for view1, view2, next_target in pretrain_loader:
            view1 = view1.float().to(device)
            view2 = view2.float().to(device)
            next_target = next_target.float().to(device)

            z1 = model.forward_backbone(view1)
            z2 = model.forward_backbone(view2)
            p1 = model.project(z1)
            p2 = model.project(z2)

            c_loss = nt_xent_loss_simple(p1, p2, temperature=TEMP)
            pred_next = model.temporal_predict(z1)
            t_loss = F.mse_loss(pred_next, next_target)
            loss = ALPHA_CONTRAST * c_loss + ALPHA_TEMPORAL * t_loss

            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(pretrain_loader)
        print(f"  Epoch {epoch + 1}/{PRETRAIN_EPOCHS} — loss: {mean_loss:.6f}")
    print("Pretraining complete.")

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    checkpoint_path = output_dir / "best_transfas_net.pth"
    epoch_records = []

    print("=== Fine-tuning ===")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        running_loss = 0.0
        for Xb, yb in finetune_train_loader:
            Xb = Xb.float().to(device)
            yb = yb.long().to(device)

            logits = model.forward_classify_from_flat(Xb)
            loss = criterion(logits, yb)
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item()

        val_acc, val_prec, val_rec, val_f1, val_roc = evaluate_classifier(
            model, finetune_valid_loader, device
        )
        print(
            f"  Epoch {epoch + 1}/{FINETUNE_EPOCHS} — "
            f"loss: {running_loss / len(finetune_train_loader):.4f}  "
            f"val_acc: {val_acc:.4f}  val_prec: {val_prec:.4f}  "
            f"val_rec: {val_rec:.4f}  val_f1: {val_f1:.4f}  val_roc: {val_roc:.4f}"
        )

        saved = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            saved = True
            print("    → Saved best model.")

        epoch_records.append({
            "epoch": epoch + 1,
            "accuracy": round(float(val_acc), 4),
            "precision": round(float(val_prec), 4),
            "recall": round(float(val_rec), 4),
            "f1": round(float(val_f1), 4),
            "roc_auc": round(float(val_roc), 4),
            "saved_checkpoint": saved,
        })

    print("Fine-tuning complete.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_acc, test_prec, test_rec, test_f1, test_roc = evaluate_classifier(
        model, finetune_test_loader, device
    )

    print("===== Test Results (v1 / TransFASNet) =====")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  F1        : {test_f1:.4f}")
    print(f"  ROC AUC   : {test_roc:.4f}")

    summary = {
        "device": str(device),
        "seed": SEED,
        "sequence_length": SEQ_LEN,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "batch_size_pretrain": BATCH_SIZE_PRETRAIN,
        "batch_size_finetune": BATCH_SIZE_FINETUNE,
        "finetune_validation_by_epoch": epoch_records,
        "best_validation_f1": round(float(best_val_f1), 4),
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall": round(float(test_rec), 4),
            "f1": round(float(test_f1), 4),
            "roc_auc": round(float(test_roc), 4),
        },
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
