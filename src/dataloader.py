import numpy as np
import torch
import spacy
from torch.utils.data import Dataset


class EnglishSpanish(Dataset):

    MAX_LEN = 16

    START_OF_SENTENCE = "<SOS>"
    END_OF_SENTENCE = "<EOS>"
    PADDING = "<PAD>"
    SPECIAL_TOKENS = [START_OF_SENTENCE, END_OF_SENTENCE, PADDING]

    def __init__(self):
        self.spacy_es = spacy.load("es_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")
        self.load_data()

    def load_data(self):
        corpus_src = self._get_tokenization("../data/sample_esp.txt", self._tokenize_es)
        corpus_tgt = self._get_tokenization("../data/sample_eng.txt", self._tokenize_en)
        # corpus_src = self._get_tokenization(
        #     "../data/gc_2010-2017_conglomerated_20171009_en.txt", self._tokenize_en
        # )
        # corpus_tgt = self._get_tokenization(
        #     "../data/gc_2010-2017_conglomerated_20171009_es.txt", self._tokenize_es
        # )

        corpus = list(zip(corpus_src, corpus_tgt))
        lens = np.array([(len(obs_src), len(obs_tgt)) for obs_src, obs_tgt in corpus])
        max_len = np.max(lens, axis=1)
        corpus = [elem for elem, num in zip(corpus, max_len) if num <= self.MAX_LEN]

        self.corpus_pre = []
        for elem_src, elem_tgt in corpus:
            inp_pad = (
                self.MAX_LEN - len(elem_src) + 1
            )  # +1 because we are adding <SOS> or <EOS> in target
            out_pad = self.MAX_LEN - len(elem_tgt)
            elem_src += [self.PADDING] * inp_pad
            elem_tgt += [self.PADDING] * out_pad
            self.corpus_pre.append((elem_src, elem_tgt, inp_pad, out_pad))

        # TODO: improve, placeholder to see if it works
        self.voc_src = self.SPECIAL_TOKENS + list(
            set(
                x
                for elem in (elem for elem, _, _, _ in self.corpus_pre)
                for x in elem
                if x not in self.SPECIAL_TOKENS
            )
        )
        self.voc_src_map = {voc: i for i, voc in enumerate(self.voc_src)}

        self.voc_tgt = self.SPECIAL_TOKENS + list(
            set(
                x
                for elem in (elem for _, elem, _, _ in self.corpus_pre)
                for x in elem
                if x not in self.SPECIAL_TOKENS
            )
        )
        self.voc_tgt_map = {voc: i for i, voc in enumerate(self.voc_tgt)}

        self.sos = self._token_to_idx([self.START_OF_SENTENCE], False).unsqueeze(1)
        self.eos = self._token_to_idx([self.END_OF_SENTENCE], False).unsqueeze(1)
        self.pad = self._token_to_idx([self.PADDING], False).unsqueeze(1)

        self.voc_src_len = len(self.voc_src_map)
        self.voc_tgt_len = len(self.voc_tgt_map)
        print("Corpus src:", self.voc_src_len)
        print("Corpus tgt:", self.voc_tgt_len)

    def _get_tokenization(self, file_pth, tokenizer):
        with open(file_pth, "r", encoding="utf8") as f:
            text = f.read().split("\n")[:-1]
            return [tokenizer(line) for line in text]

    def _tokenize_es(self, text):
        return [tok.text for tok in self.spacy_es.tokenizer(text)]

    def _tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def _token_to_idx(self, X, is_src=True):
        if is_src:
            voc_map = self.voc_src_map
        else:
            voc_map = self.voc_tgt_map
        embedding = np.zeros((len(X)))
        for i, word in enumerate(X):
            embedding[i] = voc_map[word]
        return torch.tensor(embedding, dtype=torch.long)

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
