import time

import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_and_loss(model, inp, out, inp_pad, out_pad, data, label_smooth):
    N = out.shape[0]
    c_sos = data.sos.expand(N, -1)
    c_pad = data.pad.expand(N, -1)
    # Target: end-of-sentence (EOS)
    tgt = torch.cat([out, c_pad], 1)
    tgt[range(len(out_pad)), -(out_pad + 1)] = data.eos
    tgt_oe = F.one_hot(tgt, num_classes=data.voc_tgt_len).float()
    # Input Decoder: start-of-sentence (SOS) in translation input
    out = torch.cat([c_sos, out], 1)
    # Output Decoder
    pred = model(inp.to(device), out.to(device), inp_pad, out_pad)
    J = 0
    for i in range(N):
        scale = pred.shape[1] - out_pad[i]
        t_oe = tgt_oe[i, :scale]
        p_oe = pred[i, :scale]
        # bithack to apply soft labeling (label smoothing)
        # 'label_smooth' = 0 => regular classification
        t_oe = t_oe * (1 - label_smooth) + (1 - t_oe) * label_smooth / scale
        J += F.binary_cross_entropy(p_oe.to(device), t_oe.to(device))
    return J / N


class Trainer:
    def __init__(
        self,
        iters=2,
        verbose=True,
        batch_size=64,
        lr=0.0001,
        label_smooth=False,
        shuffle=True,
    ):
        self.iters = iters
        self.verbose = verbose
        self.batch_size = batch_size
        self.lr = lr
        self.label_smooth = label_smooth
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
                train_loss = compute_and_loss(
                    model, inp, out, inp_pad, out_pad, data, self.label_smooth
                )
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item()
                print(f"{i+1}/{len(trainloader)} | loss: {train_loss.item()}")
            train_running_loss /= len(trainloader)
            train_losses.append(train_running_loss)

            if self.verbose:
                print(f"EPOCH {epoch} train loss: {train_running_loss}")
                print()

            # torch.save(model.state_dict(), "mod_checkpoint.pth")

        # Final update of the model
        if self.verbose:
            print(f"Training Finished in {(time.time()-tini)}s")
        return train_losses, valid_losses
