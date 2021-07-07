import torch
from torch.utils.data import DataLoader
from models.transformer import Transformer
from dataloader import EnglishSpanish
from train import Trainer

PICK_TOP = 0


def inference(model, inp, inp_pad, data):
    model.eval()
    sos = torch.repeat_interleave(data.sos, 1, 0)
    out1 = torch.cat([sos], 1)
    print("Initial input of the decoder")
    print(data._idx_to_token(out1))
    print()
    for _ in range(data.MAX_LEN * 2):
        R = model(inp, out1, inp_pad, [0])
        last_pred = R[0, -1]
        values, indices = torch.topk(last_pred, k=5)
        probs = values.detach().numpy()
        r = torch.tensor([indices[PICK_TOP]])
        out1 = torch.cat([out1, r.reshape(1, 1)], 1)
        print("Top best values:")
        print([(data.voc_tgt[x], p) for x, p in zip(indices, probs)])
        print()
        print("Current output:")
        print(data._idx_to_token(out1))
        print("----")
        print()
        if r == data.eos:
            break


def main():
    data = EnglishSpanish()
    model = Transformer(data.voc_src_len, data.voc_tgt_len)
    trainer = Trainer(iters=20)

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
