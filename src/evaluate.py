"""Evaluate a saved TransFASNet checkpoint on the held-out test partition."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from architecture import TransFASNet
from data import load_and_prepare_data, FinetuneDataset

SEQ_LEN = 8
EMBED_DIM = 128
TRANSFORMER_LAYERS = 3
NUM_HEADS = 4
MLP_DIM = 256
PROJ_DIM = 64
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-file", default="test_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (_, _, _, _, X_test, y_test, _, input_dim) = load_and_prepare_data(
        args.data_path, seed=SEED
    )

    model = TransFASNet(
        input_dim=input_dim, seq_len=SEQ_LEN, embed_dim=EMBED_DIM,
        n_layers=TRANSFORMER_LAYERS, n_heads=NUM_HEADS,
        mlp_dim=MLP_DIM, proj_dim=PROJ_DIM,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = DataLoader(
        FinetuneDataset(X_test, y_test), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    preds_all, probs_all, trues_all = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.float().to(device)
            logits = model.forward_classify_from_flat(Xb)
            prob = torch.softmax(logits, dim=1)
            out = torch.argmax(prob, dim=1)

            preds_all.append(out.cpu().numpy())
            probs_all.append(prob.cpu().numpy())
            trues_all.append(yb.numpy())

    preds = np.concatenate(preds_all)
    probs = np.concatenate(probs_all)
    trues = np.concatenate(trues_all)

    metrics = {
        "test_rows": int(len(trues)),
        "fraud_rows": int(trues.sum()),
        "accuracy": float((preds == trues).mean()),
        "precision": float(precision_score(trues, preds, zero_division=0)),
        "recall": float(recall_score(trues, preds, zero_division=0)),
        "f1": float(f1_score(trues, preds, zero_division=0)),
        "average_precision": float(average_precision_score(trues, probs[:, 1])),
        "roc_auc": float(roc_auc_score(trues, probs[:, 1])),
        "confusion_matrix": confusion_matrix(trues, preds).tolist(),
    }

    output_path = Path(args.output_file)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
