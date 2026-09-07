"""Shape and behavior checks for the TransFASNet model"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architecture import TransFASNet, nt_xent_loss_simple


def test_backbone_output_shape():
    model = TransFASNet(input_dim=30, seq_len=8)
    inputs = torch.randn(4, 8, 30)
    embeddings = model.forward_backbone(inputs)
    assert embeddings.shape == (4, 128)


def test_classify_from_flat_output_shape():
    model = TransFASNet(input_dim=30, seq_len=8)
    inputs = torch.randn(4, 30)
    logits = model.forward_classify_from_flat(inputs)
    assert logits.shape == (4, 2)


def test_projection_head_output_shape():
    model = TransFASNet(input_dim=30, seq_len=8, proj_dim=64)
    inputs = torch.randn(4, 8, 30)
    embeddings = model.forward_backbone(inputs)
    projected = model.project(embeddings)
    assert projected.shape == (4, 64)


def test_temporal_head_output_shape():
    model = TransFASNet(input_dim=30, seq_len=8)
    inputs = torch.randn(4, 8, 30)
    embeddings = model.forward_backbone(inputs)
    predicted_next = model.temporal_predict(embeddings)
    assert predicted_next.shape == (4, 30)


def test_classify_output_shape():
    model = TransFASNet(input_dim=30, seq_len=8)
    inputs = torch.randn(4, 8, 30)
    embeddings = model.forward_backbone(inputs)
    logits = model.classify(embeddings)
    assert logits.shape == (4, 2)


def test_nt_xent_loss_is_finite_and_positive():
    z1 = torch.randn(8, 64)
    z2 = torch.randn(8, 64)
    loss = nt_xent_loss_simple(z1, z2, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
