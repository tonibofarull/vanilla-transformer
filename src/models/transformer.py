"""
Attention Is All You Need: https://arxiv.org/abs/1706.03762

Implementation of a Vanilla Transformer



In training:

input of encoder: hello my name is toni
input of decoder: <s> hola me llamo toni
target of decoder: hola me llamo toni </s>

In inference
1r/
input of decoder: <s>
output of decoder: hola
2n/
input of decoder: <s> hola
output of decoder: hola me
...
"""

import torch
import torch.nn.functional as F
from torch import nn
from .modules import MultiHeadAttention, PositionalEncoding, Embedding


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHeadAttention()
        self.add_norm1 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.add_norm2 = nn.LayerNorm(64)

    def forward(self, X):
        X1 = self.mha(X, X, X)
        X = self.add_norm1(X1 + X)

        X1 = F.relu(self.fc1(X))
        X1 = self.fc2(X1)

        X = self.add_norm2(X1 + X)
        return X


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHeadAttention(mask=True)
        self.add_norm1 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.add_norm2 = nn.LayerNorm(64)

    def forward(self, enc, X):
        X1 = self.mha(enc, enc, X)
        X = self.add_norm1(X1 + X)

        X1 = F.relu(self.fc1(X))
        X1 = self.fc2(X1)

        X = self.add_norm2(X1 + X)
        return X


class Transformer(nn.Module):
    def __init__(self, dim_out=10, Nx=1):
        super().__init__()
        self.embedding_src = Embedding(13711, 64)
        self.embedding_tgt = Embedding(18114, 64)
        self.pe = PositionalEncoding()
        self.encs = nn.ModuleList([Encoder() for _ in range(Nx)])
        self.decs = nn.ModuleList([Decoder() for _ in range(Nx)])
        self.fc1 = nn.Linear(64, dim_out)

    def forward(self, inp, out):
        inp = self.pe(self.embedding_src(inp))
        for enc in self.encs:
            inp = enc(inp)

        out = self.pe(self.embedding_tgt(out))
        for dec in self.decs:
            out = dec(inp, out)

        X = F.softmax(self.fc1(out), dim=-1)
        return X


if __name__ == "__main__":
    tm = Transformer()
    X = [[0, 1, 0], [1, 0, 0]]
    X = torch.tensor(X, dtype=torch.float32)
    r = tm(X, X)
    print(r)
    print(r.shape)
