import time

import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader


def compute_and_loss(model, inp, out, sos, eos):
    # sos and eos shape: (1, 1)
    sos = torch.repeat_interleave(sos, out.shape[0], 0)
    eos = torch.repeat_interleave(eos, out.shape[0], 0)
    # sos and eos with shape (N, 1) and same value at every position
    # Add sos and eos to create input and output of decoder
    # out shape: (N, 16) => out1 and out2 shape (N, 17)
    out1 = torch.cat([sos, out], 1)
    out2 = torch.cat([out, eos], 1)

    # pred shape: (N, L, 18114)
    # For each position L the probabilities of all possible tokens
    pred = model(inp, out1)
    pred = pred.reshape(-1, 18114)
    real = out2.reshape(-1)
    return F.cross_entropy(pred, real)

class Trainer:
    def __init__(self):
        self.iters = 1
        self.verbose = True

    def fit(self, model, data):
        # Datasets and optimizer
        trainloader = DataLoader(data, batch_size=64, shuffle=True)
        optimizer = optim.Adam(model.parameters())
        # Training loop
        tini = time.time()
        train_losses, valid_losses = [], []

        for epoch in range(1, self.iters + 1):
            train_running_loss = 0
            model.train()
            for i, batch in enumerate(trainloader):
                optimizer.zero_grad()
                inp, out, _, _ = batch
                train_loss = compute_and_loss(model, inp, out, data.sos, data.eos)
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item()
                print(f"{i}/{len(trainloader)} | loss: {train_loss.item()}")
            train_running_loss /= len(trainloader)
            train_losses.append(train_running_loss)

            if self.verbose:
                print(f"EPOCH {epoch} train loss: {train_running_loss}")
                print()

        # Final update of the model
        if self.verbose:
            print(f"Training Finished in {(time.time()-tini)}s")
        return train_losses, valid_losses
