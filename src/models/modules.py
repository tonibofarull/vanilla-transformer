import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Embedding(nn.Module):
    """
    Converts input into embedding of correct size.
    [L, d_ini] => [L, d_model]. eg., we have 10000 and the mebedding is 512 then,
    [L, 10000] => [L, 512]. Where L is the number of tokens (words).
    """

    def __init__(self, voc_len, d_model):
        super().__init__()
        self.voc_len = voc_len
        self.fc = nn.Linear(voc_len, d_model)

    def forward(self, X):
        # print(X.shape)
        # print(self.voc_len)
        r = [self.fc(X[:, i, :]) for i in range(16)]
        X = torch.stack(r, dim=1)
        # print(X.shape)
        return X


class PositionalEncoding(nn.Module):
    """
    Section 3.5 from original paper
    """

    def __init__(self, max_seq_len=100, d_model=64):
        super().__init__()
        self.pe = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        val = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        self.pe[:, 0::2] = torch.sin(pos / val)
        self.pe[:, 1::2] = torch.cos(pos / val)

    def forward(self, X):
        # print(X.shape)
        # print(self.pe.shape)
        return X + self.pe[: X.shape[1], :]


class MultiHeadAttention(nn.Module):
    """
    Section 3.2.2 from original paper
    TODO: MULTIHEAD
    """

    def __init__(self, mask=False):
        super().__init__()
        self.mask = mask

    def forward(self, Q, K, V):
        QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.shape[1])
        if self.mask:  # block to see future elements
            mask = torch.ones((QK.shape[1], QK.shape[2]))
            mask = torch.triu(mask, diagonal=1)
            QK += mask * (-np.inf)
        att = F.softmax(QK, dim=-1)
        return torch.matmul(att, V)
