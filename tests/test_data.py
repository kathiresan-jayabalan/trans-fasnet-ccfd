"""Checks for sliding windows, augmentation, and dataset classes"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import (
    sliding_windows,
    augment_noise,
    augment_masking,
    make_view,
    PretrainDataset,
    FinetuneDataset,
)


def test_sliding_windows_shape():
    arr = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)
    windows = sliding_windows(arr, seq_len=8)
    assert windows.shape == (43, 8, 3)


def test_sliding_windows_too_short_returns_empty():
    arr = np.zeros((5, 3), dtype=np.float32)
    windows = sliding_windows(arr, seq_len=8)
    assert windows.shape == (0, 8, 3)


def test_augment_noise_changes_values():
    x = np.zeros((8, 30), dtype=np.float32)
    noisy = augment_noise(x, sigma=0.02)
    assert not np.array_equal(x, noisy)


def test_augment_masking_zeroes_some_values():
    x = np.ones((8, 30), dtype=np.float32)
    masked = augment_masking(x, mask_prob=1.0)
    assert np.all(masked == 0.0)


def test_make_view_returns_same_shape():
    x = np.random.rand(8, 30).astype(np.float32)
    view = make_view(x)
    assert view.shape == x.shape


def test_pretrain_dataset_item_shapes():
    windows = np.random.rand(10, 8, 30).astype(np.float32)
    dataset = PretrainDataset(windows)
    view1, view2, next_target = dataset[0]
    assert view1.shape == (8, 30)
    assert view2.shape == (8, 30)
    assert next_target.shape == (30,)


def test_finetune_dataset_item_shapes():
    X = np.random.rand(10, 30).astype(np.float32)
    y = np.random.randint(0, 2, size=10)
    dataset = FinetuneDataset(X, y)
    x_item, y_item = dataset[0]
    assert x_item.shape == (30,)
    assert isinstance(int(y_item), int)
