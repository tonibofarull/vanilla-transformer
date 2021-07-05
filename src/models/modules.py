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

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4 * 64)
        self.fc2 = nn.Linear(4 * 64, 64)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, max_l=100, d_model=64):
        super().__init__()
        self.pe = torch.zeros((max_l, d_model))
        pos = torch.arange(0, max_l).unsqueeze(1)
        val = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        self.pe[:, 0::2] = torch.sin(pos / val)
        self.pe[:, 1::2] = torch.cos(pos / val)

    def forward(self, X):
        return X + self.pe[: X.shape[0], :]


class MultiHeadAttention(nn.Module):
    """
    TODO: MULTIHEAD
    """
    def __init__(self, mask=False):
        super().__init__()
        self.mask = mask

    def forward(self, Q, K, V):
        QK = torch.mm(Q, K.t()) / np.sqrt(K.shape[0])
        if self.mask:  # block to see future elements
            mask = torch.ones((QK.shape[0], QK.shape[1]))
            mask = torch.triu(mask, diagonal=1)
            QK += mask * (-1e9)
        att = F.softmax(QK, dim=-1)
        return torch.mm(att, V)
