"""
Attention Is All You Need: https://arxiv.org/abs/1706.03762

Implementation of a Vanilla Transformer
"""

import torch
import torch.nn.functional as F
from torch import nn
from .modules import MultiHeadAttention, PositionalEncoding, Embedding


class Encoder(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.mha = MultiHeadAttention()
        self.add_norm1 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.add_norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        X1 = self.mha(X, X, X)
        X = self.add_norm1(X1 + X)

        X1 = F.relu(self.fc(X))
        X = self.add_norm2(X1 + X)
        return X


class Decoder(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.mha1 = MultiHeadAttention(mask=True)
        self.add_norm1 = nn.LayerNorm(d_model)
        self.mha2 = MultiHeadAttention()
        self.add_norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.add_norm3 = nn.LayerNorm(d_model)

    def forward(self, enc, X):
        X1 = self.mha1(X, X, X)
        X = self.add_norm1(X1 + X)

        X1 = self.mha2(X, enc, enc)
        X = self.add_norm2(X1 + X)

        X1 = F.relu(self.fc(X))
        X = self.add_norm3(X1 + X)
        return X


class Transformer(nn.Module):
    def __init__(self, voc_src=13711, voc_tgt=18114, d_model=64, Nx=1):
        super().__init__()
        self.embedding_src = Embedding(voc_src, d_model)
        self.embedding_tgt = Embedding(voc_tgt, d_model)
        self.pe = PositionalEncoding(d_model)
        self.encs = nn.ModuleList([Encoder() for _ in range(Nx)])
        self.decs = nn.ModuleList([Decoder() for _ in range(Nx)])
        self.fc1 = nn.Linear(d_model, voc_tgt)

    def forward(self, inp, out):
        """
        inp: (N, L)
        out: (N, L)
        """
        inp = self.embedding_src(inp)  # (N, L, d_model)
        inp = self.pe(inp)
        for enc in self.encs:
            inp = enc(inp)

        out = self.embedding_tgt(out)  # (N, L, d_model)
        out = self.pe(out)
        for dec in self.decs:
            out = dec(inp, out)

        X = self.fc1(out)  # (N, L, dim_out)
        X = F.softmax(X, dim=-1)
        return X
