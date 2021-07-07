import numpy as np
import torch
from torch.utils.data import Dataset


class SourceTargetDataset(Dataset):
    START_OF_SENTENCE = "<SOS>"
    END_OF_SENTENCE = "<EOS>"
    PADDING = "<PAD>"
    SPECIAL_TOKENS = [START_OF_SENTENCE, END_OF_SENTENCE, PADDING]

    def __init__(self, max_len, max_num_lines, option):
        self.MAX_LEN = max_len
        self.MAX_NUM_LINES = max_num_lines
        self.option = option
        self.load_data()

    def load_data(self):
        if self.option == 1:
            corpus_src = self._read_corpus_tokenized("../data/english_test.txt")
            corpus_tgt = self._read_corpus_tokenized("../data/spanish_test.txt")
        else:
            corpus_src = self._read_corpus_tokenized("../data/spanish_dataset.txt")
            corpus_tgt = self._read_corpus_tokenized("../data/english_dataset.txt")

        corpus = list(zip(corpus_src, corpus_tgt))
        lens = np.array([(len(obs_src), len(obs_tgt)) for obs_src, obs_tgt in corpus])
        max_len = np.max(lens, axis=1)
        corpus = [elem for elem, num in zip(corpus, max_len) if num <= self.MAX_LEN]

        self.corpus_pre = []
        for elem_src, elem_tgt in corpus:
            # +1 because we are adding 1 extra token in the output
            inp_pad = self.MAX_LEN - len(elem_src) + 1
            out_pad = self.MAX_LEN - len(elem_tgt)
            elem_src += [self.PADDING] * inp_pad
            elem_tgt += [self.PADDING] * out_pad
            self.corpus_pre.append((elem_src, elem_tgt, inp_pad, out_pad))

        sentences = [elem for elem, _, _, _ in self.corpus_pre]
        self.voc_src = self.SPECIAL_TOKENS.copy()
        self.voc_src += list(set(x for snts in sentences for x in snts if x not in self.SPECIAL_TOKENS))
        self.voc_src_map = dict({voc: i for i, voc in enumerate(self.voc_src)})

        sentences = [elem for _, elem, _, _ in self.corpus_pre]
        self.voc_tgt = self.SPECIAL_TOKENS.copy()
        self.voc_tgt += list(set(x for snts in sentences for x in snts if x not in self.SPECIAL_TOKENS))
        self.voc_tgt_map = dict({voc: i for i, voc in enumerate(self.voc_tgt)})

        self.sos = self._token_to_idx([self.START_OF_SENTENCE], False).unsqueeze(1)
        self.eos = self._token_to_idx([self.END_OF_SENTENCE], False).unsqueeze(1)
        self.pad = self._token_to_idx([self.PADDING], False).unsqueeze(1)

        self.voc_src_len = len(self.voc_src_map)
        self.voc_tgt_len = len(self.voc_tgt_map)
        print("Corpus src:", self.voc_src_len)
        print("Corpus tgt:", self.voc_tgt_len)

    def _read_corpus_tokenized(self, file_pth):
        with open(file_pth, "r") as f:
            r = []
            i = 0
            for x in f:
                r.append(x.rstrip().split())
                i += 1
                if i == self.MAX_NUM_LINES:
                    break
            return r

    def _token_to_idx(self, X, is_src=True):
        if is_src:
            voc_map = self.voc_src_map
        else:
            voc_map = self.voc_tgt_map
        idxs = np.zeros((len(X)))
        for i, word in enumerate(X):
            idxs[i] = voc_map[word]
        return torch.tensor(idxs, dtype=torch.long)

    def _idx_to_token(self, X, is_src=False):
        voc = self.voc_tgt
        if is_src:
            voc = self.voc_src
        r = []
        for batch in X:
            tra = []
            for res in batch:
                tra.append(voc[res])
            r.append(tra)
        return r

    def __len__(self):
        return len(self.corpus_pre)

    def __getitem__(self, idx):
        elem_src, elem_tgt, inp_pad, out_pad = self.corpus_pre[idx]
        return (
            self._token_to_idx(elem_src),
            self._token_to_idx(elem_tgt, False),
            inp_pad,
            out_pad,
            elem_src,
            elem_tgt,
        )
