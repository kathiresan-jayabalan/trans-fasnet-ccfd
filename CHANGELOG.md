# Changelog

## 1.0.0 (Initial release)

- Added TransFASNet architecture featuring a single Transformer encoder with contrastive pretraining (NT-Xent) and an auxiliary next-transaction forecasting head.
- Added random stratified 80/10/10 train, validation, and test split preserving class distribution.
- Added global feature scaling via MinMaxScaler.
- Added BorderlineSMOTE oversampling exclusively on the training partition for class balancing during fine-tuning.
- Added sliding-window construction for pretraining from the scaled feature matrix.
- Added evaluation script reporting accuracy, precision, recall, F1, average precision, ROC-AUC, and confusion matrix.
- Added a comprehensive unit and integration test suite covering model shapes, data utilities, and end-to-end execution.
