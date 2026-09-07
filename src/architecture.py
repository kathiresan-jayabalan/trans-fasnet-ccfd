"""TransFASNet model definition"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransFASNet(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        embed_dim=128,
        n_layers=3,
        n_heads=4,
        mlp_dim=256,
        proj_dim=64,
        num_classes=2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=mlp_dim, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward_backbone(self, x):
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return h.mean(dim=1)

    def project(self, z):
        return self.proj_head(z)

    def temporal_predict(self, z):
        return self.temporal_head(z)

    def classify(self, z):
        return self.classifier(z)

    def forward_classify_from_flat(self, x_flat):
        z = self.embed(x_flat.unsqueeze(1))
        z = self.pos_enc(z)
        z = self.transformer(z)
        return self.classify(z.mean(dim=1))


def nt_xent_loss_simple(z1, z2, temperature=0.1):
    """NT-Xent contrastive loss (InfoNCE)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(reps, reps.T)
    mask = (~torch.eye(2 * B, dtype=torch.bool, device=z1.device)).float()
    exp_sim = torch.exp(sim / temperature) * mask
    denom = exp_sim.sum(dim=1)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)]) / temperature
    return (-(pos - torch.log(denom))).mean()


if __name__ == "__main__":
    model = TransFASNet(input_dim=30, seq_len=8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransFASNet (v1) — {params:,} trainable parameters")
    x = torch.randn(4, 8, 30)
    z = model.forward_backbone(x)
    print(f"Backbone output: {z.shape}")
