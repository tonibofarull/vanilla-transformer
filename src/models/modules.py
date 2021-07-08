import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    """
    Index of a word to the embedding representation
    """

    def __init__(self, voc_len, d_model):
        super().__init__()
        self.table = nn.Embedding(voc_len, d_model)

    def forward(self, X):
        X = self.table(X)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        pe = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        val = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(pos / val)
        pe[:, 1::2] = torch.cos(pos / val)
        # save buffer in state_dict but not trained by optimizer
        self.register_buffer("pe", pe)

    def forward(self, X):
        X = X + self.pe[: X.shape[1], :]
        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, is_mask):
        super().__init__()
        self.is_mask = is_mask

        d = d_model // h
        self.fc_q = nn.ModuleList([nn.Linear(d_model, d) for _ in range(h)])
        self.fc_k = nn.ModuleList([nn.Linear(d_model, d) for _ in range(h)])
        self.fc_v = nn.ModuleList([nn.Linear(d_model, d) for _ in range(h)])
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, pad):
        """
        :param pad: list containing the padding of each element of the batch.
        """
        Qs = [proj(Q) for proj in self.fc_q]
        Ks = [proj(K) for proj in self.fc_k]
        Vs = [proj(V) for proj in self.fc_v]
        X = [self._SDA(Q, K, V, pad) for Q, K, V in zip(Qs, Ks, Vs)]
        X = torch.cat(X, dim=-1)
        X = self.fc(X)
        return X

    def _SDA(self, Q, K, V, pad):
        """
        Scaled Dot-Product Attention
        """
        QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.shape[1])
        M = torch.zeros(QK.shape).to(device)
        for i, x in enumerate(pad):
            M[i, :, -x:] = 1
            # M[i, -x:, :] = 1
        if self.is_mask:  # to predict word i, mask positions j such that j > i
            future_M = torch.ones((QK.shape[1], QK.shape[2])).to(device)
            future_M = torch.triu(future_M, diagonal=1)
            M = torch.maximum(M, future_M)
        QK = QK + M * (-1e9)
        att = F.softmax(QK, dim=-1)
        X = torch.matmul(att, V)
        return X
