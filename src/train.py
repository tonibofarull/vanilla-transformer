import time

import torch
from torch import optim
from torch.utils.data import DataLoader

def compute_loss(pred, real):
    return torch.sum(pred)

class Trainer:
    def __init__(self):
        self.iters = 10
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
                print(f"{i}/{len(trainloader)}")
                optimizer.zero_grad()
                inp, out = batch
                train_loss = compute_loss(model(inp, out), out)
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item()

            train_running_loss /= len(trainloader)
            train_losses.append(train_running_loss)

            if self.verbose:
                print(f"EPOCH {epoch} train loss: {train_running_loss}")
                print()

        # Final update of the model
        if self.verbose:
            print(f"Training Finished in {(time.time()-tini)}s")
        return train_losses, valid_losses
