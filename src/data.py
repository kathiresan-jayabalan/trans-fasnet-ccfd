"""Data loading, preprocessing, augmentation, and dataset classes"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import BorderlineSMOTE


def load_and_prepare_data(file_path, seed=42):
    """Loads the dataset, scales features, splits 80/10/10, and applies
    BorderlineSMOTE to the training partition.

    Returns X_train_over, y_train_over, X_valid, y_valid, X_test, y_test,
    X_scaled, and input_dim. X_scaled (the full scaled feature matrix) is
    returned separately because pretraining windows are built from it
    before the split, matching the original notebook.
    """
    df = pd.read_csv(file_path)
    df_numeric = df.select_dtypes(include=['number']).copy()

    if 'Time' not in df_numeric.columns:
        df_numeric['Time'] = np.arange(len(df_numeric))

    df_numeric = df_numeric.sort_values('Time').reset_index(drop=True)

    y = df_numeric['Class'].values
    X = df_numeric.drop('Class', axis=1).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, shuffle=True, stratify=y_temp
    )

    sm = BorderlineSMOTE(kind='borderline-1', random_state=seed)
    X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

    input_dim = X_train.shape[1]

    return X_train_over, y_train_over, X_valid, y_valid, X_test, y_test, X_scaled, input_dim


def sliding_windows(arr, seq_len):
    N = arr.shape[0]
    if N < seq_len + 1:
        return np.zeros((0, seq_len, arr.shape[1]))
    return np.array([arr[i:i + seq_len] for i in range(0, N - seq_len + 1)])


def augment_noise(x, sigma=0.02):
    return x + np.random.normal(0, sigma, size=x.shape)


def augment_masking(x, mask_prob=0.15):
    x2 = x.copy()
    mask = np.random.rand(*x2.shape) < mask_prob
    x2[mask] = 0.0
    return x2


def make_view(x):
    x_aug = augment_noise(x, sigma=0.02)
    x_aug = augment_masking(x_aug, mask_prob=0.12)
    return x_aug


class PretrainDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows.astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq = self.windows[idx]
        view1 = make_view(seq)
        view2 = make_view(seq)
        next_target = seq[-1].copy()
        return view1, view2, next_target


class FinetuneDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
