import torch
import torch.nn.functional as F
from torch import nn
from .modules import MultiHeadAttention, PositionalEncoding, Embedding


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, drop_p, h):
        super().__init__()
        self.mha = MultiHeadAttention(is_mask=False, d_model=d_model, h=h)
        self.drop1 = nn.Dropout(drop_p)
        self.batch_norm1 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(drop_p)
        self.batch_norm2 = nn.LayerNorm(d_model)

    def forward(self, X, pad):
        X1 = self.mha(X, X, X, pad)
        X1 = self.drop1(X1)
        X = self.batch_norm1(X + X1)

        X1 = F.relu(self.fc1(X))
        X1 = self.fc2(X1)
        X1 = self.drop2(X1)
        X = self.batch_norm2(X + X1)
        return X


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, drop_p, h):
        super().__init__()
        self.mha1 = MultiHeadAttention(is_mask=True, d_model=d_model, h=h)
        self.drop1 = nn.Dropout(drop_p)
        self.batch_norm1 = nn.LayerNorm(d_model)

        self.mha2 = MultiHeadAttention(is_mask=True, d_model=d_model, h=h)
        self.drop2 = nn.Dropout(drop_p)
        self.batch_norm2 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop3 = nn.Dropout(drop_p)
        self.batch_norm3 = nn.LayerNorm(d_model)

    def forward(self, X, Y, pad):
        Y1 = self.mha1(Y, Y, Y, pad)
        Y1 = self.drop1(Y1)
        Y = self.batch_norm1(Y + Y1)

        Y1 = self.mha2(Y, X, X, pad)
        Y1 = self.drop2(Y1)
        Y = self.batch_norm2(Y + Y1)

        Y1 = F.relu(self.fc1(Y))
        Y1 = self.fc2(Y1)
        Y1 = self.drop3(Y1)
        Y = self.batch_norm3(Y + Y1)
        return Y


class Transformer(nn.Module):
    def __init__(self, voc_src, voc_tgt, d_model=512, d_ff=2048, drop_p=0.1, Nx=6, h=8):
        super().__init__()
        self.pe = PositionalEncoding(d_model)

        self.embedding_src = Embedding(voc_src, d_model)
        self.drop1 = nn.Dropout(drop_p)
        self.encs = nn.ModuleList(
            [Encoder(d_model, d_ff, drop_p, h) for _ in range(Nx)]
        )

        self.embedding_tgt = Embedding(voc_tgt, d_model)
        self.drop2 = nn.Dropout(drop_p)
        self.decs = nn.ModuleList(
            [Decoder(d_model, d_ff, drop_p, h) for _ in range(Nx)]
        )

        self.fc = nn.Linear(d_model, voc_tgt)

    def forward(self, inp, out, inp_pad, out_pad):
        """
        :param inp: sentence to translate, shape (N, L1)
        :param out: tokens to translate, shape (N, L2).
        In training L1=L2, in inference L2 is the number of words to translate in an autoregressive mode.
        :return: shape (N, L, voc_tgt)
        """
        # Encoder
        inp = self.embedding_src(inp)  # (N, L1, voc_src)
        inp = self.pe(inp)
        inp = self.drop1(inp)
        for enc in self.encs:
            inp = enc(inp, inp_pad)

        # Decoder
        out = self.embedding_tgt(out)  # (N, L1, voc_tgt)
        out = self.pe(out)
        out = self.drop2(out)
        for dec in self.decs:
            out = dec(inp, out, out_pad)

        # Output
        X = self.fc(out)
        X = F.softmax(X, dim=-1)
        return X
