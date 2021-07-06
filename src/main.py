import torch
from torch.nn import functional as F
import numpy as np
import seaborn as sns
from models.transformer import Transformer
from train import Trainer
from dataloader import EnglishToSpanish
from torch.utils.data import DataLoader

def main():
    trainer = Trainer()
    tm = Transformer()
    data = EnglishToSpanish()
    dl = iter(DataLoader(data))

    # trainer.fit(tm, data)

    inp, out, src, tgt = next(dl)
    sos = torch.repeat_interleave(data.sos, 1, 0)
    out1 = torch.cat([sos], 1)
    R = tm(inp, out1)
    last_pred = R[0,-1]
    print(f"Adding with prob. {torch.max(last_pred)}")
    r = torch.argmax(last_pred)
    out1 = torch.cat([out1, r.reshape(1, 1)], 1)
    print(out1)


if __name__ == "__main__":
    main()
