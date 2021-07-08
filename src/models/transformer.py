import torch.nn.functional as F
from torch import nn
from .modules import MultiHeadAttention, PositionalEncoding, Embedding


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p, postLN):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(drop_p)
        self.norm = nn.LayerNorm(d_model)
        self.postLN = postLN

    def forward(self, X):
        if self.postLN:
            X1 = F.relu(self.fc1(X))
            X1 = self.fc2(X1)
            X1 = self.drop(X1)
            X = self.norm(X + X1)
        else:
            X1 = self.norm(X)
            X1 = F.relu(self.fc1(X1))
            X1 = self.fc2(X1)
            X = X + self.drop(X1)
        return X


class Block(nn.Module):
    def __init__(self, d_model, h, drop_p, postLN, is_mask):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, h=h, is_mask=is_mask)
        self.drop = nn.Dropout(drop_p)
        self.norm = nn.LayerNorm(d_model)
        self.postLN = postLN

    def forward(self, Q, K, V, pad):
        if self.postLN:
            X = self.mha(Q, K, V, pad)
            X = self.drop(X)
            X = self.norm(Q + X)
        else:
            Q = self.norm(Q)
            X = self.mha(Q, K, V, pad)
            X = Q + self.drop(X)
        return X


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, drop_p, h, postLN):
        super().__init__()
        self.block = Block(d_model, h, drop_p, postLN, is_mask=False)
        self.ff = FeedForward(d_model, d_ff, drop_p, postLN)
        self.postLN = postLN
        if not self.postLN:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, X, pad):
        X = self.block(X, X, X, pad)
        X = self.ff(X)
        if not self.postLN:
            X = self.norm(X)
        return X


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, drop_p, h, postLN):
        super().__init__()
        self.block1 = Block(d_model, h, drop_p, postLN, is_mask=True)
        self.block2 = Block(d_model, h, drop_p, postLN, is_mask=True)
        self.ff = FeedForward(d_model, d_ff, drop_p, postLN)
        self.postLN = postLN
        if not self.postLN:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, X, Y, pad):
        Y = self.block1(Y, Y, Y, pad)
        Y = self.block2(Y, X, X, pad)
        Y = self.ff(Y)
        if not self.postLN:
            X = self.norm(X)
        return Y


class Transformer(nn.Module):
    def __init__(self, voc_src, voc_tgt, d_model=512, d_ff=2048, drop_p=0.1, Nx=6, h=8, postLN=True):
        super().__init__()
        self.pe = PositionalEncoding(d_model)

        self.embedding_src = Embedding(voc_src, d_model)
        self.drop1 = nn.Dropout(drop_p)
        self.encs = nn.ModuleList(
            [Encoder(d_model, d_ff, drop_p, h, postLN) for _ in range(Nx)]
        )

        self.embedding_tgt = Embedding(voc_tgt, d_model)
        self.drop2 = nn.Dropout(drop_p)
        self.decs = nn.ModuleList(
            [Decoder(d_model, d_ff, drop_p, h, postLN) for _ in range(Nx)]
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
