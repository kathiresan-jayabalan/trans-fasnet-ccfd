# TransFASNet for Credit Card Fraud Detection (trans-fasnet-ccfd)

This repository contains a baseline implementation of TransFASNet for binary credit card fraud classification.

The model uses a single Transformer encoder to process fixed-length transaction windows. During pretraining, two augmented views of each transaction window are used for contrastive learning, together with an auxiliary next-step prediction task. The pretrained model is then fine-tuned for fraud classification.

## Model

- Input: Transaction windows of length 8
- Encoder: 3-layer Transformer with 4 attention heads
- Embedding size: 128
- Pretraining: NT-Xent contrastive loss with an auxiliary next-transaction prediction loss (mean-squared-error)
- Fine-tuning: Cross-entropy loss for binary fraud classification

## Data

The project uses the Kaggle Credit Card Fraud Detection dataset.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File name: `creditcard.csv`
- Target column: `Class`
- Time column: `Time`

Download the dataset and place it at `data/creditcard.csv`.

## Data split

The records are sorted by `Time` before the dataset is split.

- Training: first 70 %
- Validation: next 10 %
- Test: final 20 %

The scaler is fitted using the training partition only. Transaction windows are then created separately for the training, validation and test partitions.

## Sequence label

Each sequence contains 8 consecutive transactions. The label assigned to a sequence is the class of its final transaction.

## Running the Experiments

Install the required dependencies:
```bash
pip install -r requirements.txt

Train the model:
```bash
python src/train.py --data-path data/creditcard.csv --output-dir outputs

Evaluate the trained model:
```bash
python src/evaluate.py --data-path data/creditcard.csv --checkpoint outputs/best_model.pt
