import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Embedding(nn.Module):
    """
    Converts input into embedding of correct size.
    """

    def __init__(self, voc_len, d_model):
        super().__init__()
        self.table = nn.Embedding(voc_len, d_model)

    def forward(self, X):
        """
        X:      (N, L)
        return: (N, L, d_model)
        """
        X = self.table(X)
        return X


class PositionalEncoding(nn.Module):
    """
    Section 3.5 from original paper
    """

    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        pe = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        val = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(pos / val)
        pe[:, 1::2] = torch.cos(pos / val)
        # to not consider the buffer as a parameter
        self.register_buffer("pe", pe)

    def forward(self, X):
        """
        X:      (N, L, d_model)
        return: (N, L, d_model)
        """
        X = X + self.pe[: X.shape[1], :]
        return X


class MultiHeadAttention(nn.Module):
    """
    Section 3.2.2 from original paper
    TODO: We are using singlehead attention right now. DO: MULTIHEAD
    TODO: proper masking
    """

    def __init__(self, d_model=64, h=8, mask=True):
        super().__init__()
        self.h = h
        self.mask = mask
        # d = d_model // h
        # self.fc_q = nn.ModuleList([nn.Linear(d, d) for _ in range(h)])
        # self.fc_k = nn.ModuleList([nn.Linear(d, d) for _ in range(h)])
        # self.fc_v = nn.ModuleList([nn.Linear(d, d) for _ in range(h)])
        # self.fc = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        return self._SDA(Q, K, V)

    def _SDA(self, Q, K, V):
        """
        Scaled Dot-Product Attention: Section 3.2.1
        """
        QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.shape[1])
        if self.mask:  # block to see future elements
            mask = torch.ones((QK.shape[1], QK.shape[2]))
            mask = torch.triu(mask, diagonal=1)
            QK += mask * (-1e9)
        att = F.softmax(QK, dim=-1)
        X = torch.matmul(att, V)
        return X
