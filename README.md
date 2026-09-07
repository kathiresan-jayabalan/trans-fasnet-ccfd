# TransFASNet for Credit Card Fraud Detection (trans-fasnet-ccfd)

This repository contains a baseline implementation of TransFASNet for binary credit card fraud classification.

The model uses a single Transformer encoder to process fixed-length transaction windows. Pretraining combines two objectives: an NT-Xent contrastive loss between two augmented views of each window and an auxiliary loss that forecasts the transaction immediately following the window. That forecast target comes from outside the window, so the encoder can't just copy it from its own input. Fine-tuning then trains the pretrained encoder for fraud classification with cross-entropy loss, using BorderlineSMOTE on the training partition to handle class imbalance.

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan

## Originality & Novelty 

This implementation adapts contrastive self-supervised pretraining (NT-Xent / InfoNCE) to tabular financial transaction sequences, a technique typically restricted to computer vision (SimCLR, MoCo) and natural language processing. The model learns fraud-discriminative representations from unlabeled transaction windows prior to supervised fine-tuning on the imbalanced label set.

## Model Architecture

The model processes input tensors of shape $(B, T=8, D=30)$ through a single Transformer encoder that attends across the temporal dimension.

```
Input (B, T=8, D=30)
    │
    ▼
Linear(D → embed_dim=128)          ← project each timestep to embedding space
    │
PositionalEncoding (sinusoidal)
    │
TransformerEncoder                 ← 3 layers, 4 heads, GELU, d_ff=256
  [MultiheadAttention + FFN] × 3   ← attends across T timestep tokens
    │
mean pool over T
    │
    z  (B, 128)
   /│\
  / │ \
 ▼  ▼  ▼
proj  temporal  classifier
head   head
(NT-Xent) (MSE)  (CrossEntropy)
```

- **Input Projection:** Maps raw feature dimensions to the transformer embedding space.
- **Positional Encoding:** Adds sinusoidal sequence order information.
- **Shared Transformer Encoder:** Aggregates temporal context using multi-head self-attention across the $T$ timesteps.
- **Multi-Task Heads:** 
  - *Projection Head:* Optimizes NT-Xent contrastive loss between augmented views.
  - *Temporal Head:* Optimizes mean-squared error (MSE) for next-step transaction forecasting.
  - *Classifier Head:* Outputs binary fraud logits using cross-entropy.

## Model Hyperparameters

- Input: transaction windows of length 8 (configurable)
- Encoder: 3-layer Transformer with 4 attention heads (configurable)
- Embedding size: 128 (configurable)
- Pretraining: NT-Xent contrastive loss plus a mean-squared-error forecasting loss against the next real transaction
- Fine-tuning: cross-entropy loss for binary fraud classification on oversampled data

## Data

The project uses the Kaggle Credit Card Fraud Detection dataset.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File name: `creditcard.csv`
- Target column: `Class`
- Time column: `Time`

Download the dataset and place it at `data/creditcard.csv`.

## Data Split & Preprocessing

- **Split Protocol:** Random stratified 80/10/10 split (80% training, 10% validation, 10% test) preserving the class distribution across partitions.
- **Scaling:** `MinMaxScaler` is applied to the numeric feature matrix.
- **Class Balancing:** `BorderlineSMOTE` (kind='borderline-1') is applied exclusively to the training partition to address class imbalance prior to fine-tuning.
- **Pretraining Windows:** Sliding windows for contrastive and temporal pretraining are constructed from the dataset.

## Sequence Label

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
python src/evaluate.py --data-path data/creditcard.csv --checkpoint outputs/best_transfas_net.pth --output-file outputs/test_metrics.json
```

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

The suite covers random stratified splitting and scaling, window and forecasting-target construction, model output shapes, the NT-Xent loss (including a gradient-flow check) and an end-to-end run of train.py followed by evaluate.py on a dataset.

## Output
Training writes outputs/best_transfas_net.pth and outputs/training_summary.json. Evaluation writes outputs/test_metrics.json with precision, recall, F1, average precision, ROC-AUC, and a confusion matrix for the fraud class.

## Results
See [`results/README.md`](results/README.md) for the full training log, structured metrics, and executed notebook from the protocol run: 10 pretraining epochs (556 batches each, loss dropping from 4.585835 to 1.446875), 10 fine-tuning epochs (1,777 batches each) with per-epoch validation accuracy/precision/recall/F1/ROC-AUC, the checkpoint-save events and the final test block. Same way, hyperparameters (batch sizes 512/256, learning rates 3e-4/1e-4, seed 42, sequence length 8), per-epoch arrays for both pretraining loss and fine-tuning validation metrics, the best-checkpoint marker (epoch 9, F1 0.7257) and the final test metrics as shown below:

| Metric | Value |
|---|---|
| Accuracy | 0.9988 |
| Precision | 0.6212 |
| Recall | 0.8367 |
| F1 | 0.7130 |
| ROC AUC | 0.9473 |

## Status
[kathiresan-jayabalan/trans-fasnet-ccfd](https://github.com/kathiresan-jayabalan/trans-fasnet-ccfd) is a baseline implementation. It does not include spatial-feature attention or gated fusion. Because a single encoder attends over the $T$ timestep dimension treating each transaction vector as an opaque point, this baseline cannot separately isolate spatial feature anomalies from temporal pattern anomalies. That specific architectural limitation is resolved in follow-up systems, which are implemented in separate repositories: [kathiresan-jayabalan/sttn-cp-ccfd](https://github.com/kathiresan-jayabalan/sttn-cp-ccfd) and [kathiresan-jayabalan/c-sten-ccfd](https://github.com/kathiresan-jayabalan/c-sten-ccfd).
