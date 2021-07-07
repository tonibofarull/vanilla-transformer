import time

import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_and_loss(model, inp, out, inp_pad, out_pad, data):
    # sos and eos shape: (1, 1)
    c_sos = torch.repeat_interleave(data.sos, out.shape[0], 0)
    c_pad = torch.repeat_interleave(data.pad, out.shape[0], 0)
    # sos and eos with shape (N, 1) and same value at every position
    # Add sos and eos to create input and output of decoder
    # out shape: (N, 16) => out1 and out2 shape (N, 17)
    out1 = torch.cat([c_sos, out], 1)
    real = torch.cat([out, c_pad], 1)
    real[range(len(out_pad)), -(out_pad + 1)] = data.eos
    real = F.one_hot(real, num_classes=data.voc_tgt_len).float()

    pred = model(inp.to(device), out1.to(device), inp_pad, out_pad)
    r = 0
    for i in range(real.shape[0]):
        scale = real.shape[1] - out_pad[i]
        r_oe = real[i, :scale]
        p_oe = pred[i, :scale]
        # bithack to apply softlabeling
        max_p = 0.99
        r_oe = r_oe * max_p + (1 - r_oe) * (1 - max_p) / r_oe.shape[1]
        r += F.binary_cross_entropy(p_oe.to(device), r_oe.to(device)) / scale
    return r


class Trainer:
    def __init__(self, iters=2, verbose=True, batch_size=128, lr=0.0001, shuffle=True):
        self.iters = iters
        self.verbose = verbose
        self.batch_size = batch_size
        self.lr = lr
        self.shuffle = shuffle

    def fit(self, model, data):
        # Datasets and optimizer
        trainloader = DataLoader(data, batch_size=self.batch_size, shuffle=self.shuffle)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Training loop
        tini = time.time()
        train_losses, valid_losses = [], []

        for epoch in range(1, self.iters + 1):
            train_running_loss = 0
            model.train()
            for i, batch in enumerate(trainloader):
                optimizer.zero_grad()
                inp, out, inp_pad, out_pad, _, _ = batch
                train_loss = compute_and_loss(model, inp, out, inp_pad, out_pad, data)
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item()
                print(f"{i}/{len(trainloader)} | loss: {train_loss.item()}")
            train_running_loss /= len(trainloader)
            train_losses.append(train_running_loss)

            if self.verbose:
                print(f"EPOCH {epoch} train loss: {train_running_loss}")
                print()

            torch.save(model.state_dict(), "mod_checkpoint.pth")

        # Final update of the model
        if self.verbose:
            print(f"Training Finished in {(time.time()-tini)}s")
        return train_losses, valid_losses
