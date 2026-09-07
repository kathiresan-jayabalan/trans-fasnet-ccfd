# Results

This directory holds evaluation output of the model's experiment results.

## Experiment Run

This directory contains the executed run of TransFASNet using the protocol: 
random stratified 80/10/10 split, BorderlineSMOTE applied to the training partition and 
contrastive-pretraining windows built from the full scaled dataset before the split.

### Files

| File | Contents |
|---|---|
| `run_log.txt` | Full training and evaluation log |
| `metrics.json` | hyperparameters, per-epoch pretraining loss, per-epoch fine-tuning validation metrics, best checkpoint and final test metrics |
| `notebook.ipynb` | The executed Colab notebook |

### Final test results

| Metric | Value |
|---|---|
| Accuracy | 0.9988 |
| Precision | 0.6212 |
| Recall | 0.8367 |
| F1 | 0.7130 |
| ROC AUC | 0.9473 |

### Protocol

- Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions, 492 fraud, 30 features)
- Split: random, stratified by class, 80% train / 10% validation / 10% test
- Class imbalance handling: BorderlineSMOTE on the training partition
- Sequence length: 8
- Pretraining: 10 epochs, batch size 512, 556 batches per epoch, learning rate 3e-4, NT-Xent temperature 0.1
- Fine-tuning: 10 epochs, batch size 256, 1,777 batches per epoch, learning rate 1e-4
- Model selection: checkpoint saved whenever validation F1 improved and best was epoch 9 (F1 = 0.7257)
- Final evaluation: best checkpoint evaluated once on the held-out test partition

### Reproducing this run

1. Open `notebook.ipynb` in Google Colab.
2. Set the runtime to a GPU accelerator.
3. Upload `creditcard.csv` to Google Drive at `My Drive/credit card /creditcard.csv`, or update `FILE_PATH` in the notebook to your own path.
4. Install `imbalanced-learn` if it is not already available in the runtime.
5. Run all cells in order.

Random seed 42 is set for NumPy and PyTorch, so results should be close to
the values recorded here, though exact reproducibility across different
GPU hardware and library versions.
