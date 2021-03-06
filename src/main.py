import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import yaml
from models.transformer import Transformer
from dataloader import SourceTargetDataset
from train import Trainer

torch.manual_seed(4444)
np.random.seed(4444)
random.seed(4444)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PICK_TOP = 0


def inference(model, inp, inp_pad, data):
    model.eval()
    c_sos = data.sos.expand(1, -1)
    out = torch.cat([c_sos], 1)
    print("Initial input of the decoder")
    print(data._idx_to_token(out))
    print()
    for _ in range(data.max_len * 2):
        R = model(inp.to(device), out.to(device), inp_pad, [0])
        last_pred = R[0, -1]
        values, indices = torch.topk(last_pred, k=5)
        probs = values.cpu().detach().numpy()
        r = torch.tensor([indices[PICK_TOP]])
        out = torch.cat([out, r.reshape(1, 1)], 1)
        print("Top best values:")
        print([(data.voc_tgt[x], p) for x, p in zip(indices, probs)])
        print()
        print("Current output:")
        print(data._idx_to_token(out))
        print("----")
        print()
        if r == data.eos:
            break


def main():
    with open("configs/config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    data = SourceTargetDataset(**cfg["dataset"])
    model = Transformer(data.voc_src_len, data.voc_tgt_len, **cfg["model"])

    # model.load_state_dict(torch.load("mod_checkpoint.pth"))
    model.to(device)

    trainer = Trainer(**cfg["train"])
    trainer.fit(model, data)
    print()

    dl = iter(DataLoader(data, batch_size=1))

    i = 0
    for inp, out, inp_pad, out_pad, src, tgt in dl:
        print("################")
        print("Sentence to translate:")
        print(data._idx_to_token(inp, is_src=True))
        print("Ground truth:")
        print(data._idx_to_token(out))
        print()
        inference(model, inp, inp_pad, data)
        i += 1
        if i == 5:
            break


if __name__ == "__main__":
    main()
