# TransFASNet for Credit Card Fraud Detection (trans-fasnet-ccfd)

This repository contains a baseline implementation of TransFASNet for binary credit card fraud classification.

The model uses a single Transformer encoder to process fixed-length transaction windows. Pretraining combines two objectives: an NT-Xent contrastive loss between two augmented views of each window and an auxiliary loss that forecasts the transaction immediately following the window. That forecast target comes from outside the window, so the encoder can't just copy it from its own input. Fine-tuning then trains the pretrained encoder for fraud classification with class-weighted cross-entropy loss.

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan
  
## Model

- Input: transaction windows of length 8 (configurable)
- Encoder: 3-layer Transformer with 4 attention heads (configurable)
- Embedding size: 128 (configurable)
- Pretraining: NT-Xent contrastive loss plus a mean-squared-error forecasting loss against the next real transaction
- Fine-tuning: class-weighted cross-entropy for binary fraud classification, with an F1-optimal decision threshold selected on the validation partition each epoch

## Data

The project uses the Kaggle Credit Card Fraud Detection dataset.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File name: `creditcard.csv`
- Target column: `Class`
- Time column: `Time`

Download the dataset and place it at `data/creditcard.csv`.

## Data split

Records are sorted by `Time` before splitting.

- Training: first 70%
- Validation: next 10%
- Test: final 20%

`MinMaxScaler` is fit on the training partition only and applied unchanged to validation and test.

## Handling the `Time` column

A chronological split means every validation and test timestamp is larger than any timestamp seen while fitting the scaler, so the raw `Time` column can fall outside the training scaler's range at evaluation time. `--time-feature` controls this:

- `delta` (default): replace `Time` with the gap since the previous transaction. This is roughly stationary, so validation/test values stay close to the range seen during training.
- `drop`: remove `Time` entirely.
- `raw`: keep the original, unbounded, monotonically increasing column (not recommended; kept for comparison against earlier baselines).

## Sequence label

Each labeled sequence spans `sequence_length` consecutive transactions. The label assigned to the sequence is the class of its final transaction.

## Running the experiments

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python src/train.py --data-path data/creditcard.csv --output-dir outputs
```

Evaluate the trained model:
```bash
python src/evaluate.py --data-path data/creditcard.csv --checkpoint outputs/best_model.pt
```

Useful training flags (all optional, defaults shown):
```bash
python src/train.py \
  --data-path data/creditcard.csv \
  --output-dir outputs \
  --sequence-length 8 \
  --embedding-dim 128 \
  --layers 3 \
  --heads 4 \
  --feedforward-dim 256 \
  --projection-dim 64 \
  --dropout 0.1 \
  --time-feature delta \
  --batch-size 256 \
  --pretrain-epochs 15 \
  --finetune-epochs 25 \
  --learning-rate 1e-3 \
  --temperature 0.1 \
  --noise-std 0.02 \
  --mask-prob 0.12 \
  --patience 8 \
  --seed 42
```

The checkpoint saved at `outputs/best_model.pt` stores the full model configuration alongside the weights, so `evaluate.py` reconstructs the exact same architecture without needing any flags beyond `--data-path` and `--checkpoint`.

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

The suite covers chronological splitting and scaling, window and forecasting-target construction, model output shapes, the NT-Xent loss (including a gradient-flow check), and an end-to-end run of `train.py` followed by `evaluate.py` on a dataset.

## Output
Training writes outputs/best_model.pt and outputs/training_summary.json. Evaluation writes test_metrics.json with precision, recall, F1, average precision, ROC-AUC, and a confusion matrix for the fraud class.

## Results

See [`results/README.md`](results/README.md) for the full training log, structured metrics, and executed notebook from the original protocol run:

| Metric | Value |
|---|---|
| Accuracy | 0.9988 |
| Precision | 0.6212 |
| Recall | 0.8367 |
| F1 | 0.7130 |
| ROC AUC | 0.9473 |

## Status
This is a baseline implementation. It does not include spatial-feature attention or gated fusion. Those components are implemented in separate repositories:
- STTN-CP: https://github.com/kathiresan-jayabalan/sttn-cp-ccfd
- C-STEN: https://github.com/kathiresan-jayabalan/c-sten-ccfd
