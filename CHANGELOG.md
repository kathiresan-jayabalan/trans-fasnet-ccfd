# Changelog

## 1.0.0 (Initial release)

- Added TransFASNet architecture: single Transformer encoder with
  contrastive pretraining and an auxiliary next-transaction prediction
  head.
- Added chronological train, validation, and test split based on the
  `Time` column.
- Added train-only feature scaling.
- Added sequence-window construction performed separately within each
  data partition.
- Added weighted cross-entropy for the fine-tuning stage.
- Added evaluation script reporting fraud-class precision, recall, F1,
  average precision, ROC-AUC and confusion matrix.
- Added unit tests for model output shapes and window construction.
