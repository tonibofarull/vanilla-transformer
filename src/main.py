from pydantic.errors import DataclassTypeError
import torch
from torch.nn import functional as F
import numpy as np
import seaborn as sns
from models.transformer import Transformer
from train import Trainer
from dataloader import EnglishToSpanish


def main():
    trainer = Trainer()
    tm = Transformer()
    data = EnglishToSpanish()

    trainer.fit(tm, data)


if __name__ == "__main__":
    main()
