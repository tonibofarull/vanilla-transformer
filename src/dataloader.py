import numpy as np
import torch
import spacy
from torch.utils.data import DataLoader, Dataset


class EnglishToSpanish(Dataset):

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
        corpus_src = self._get_tokenization(
            "../data/gc_2010-2017_conglomerated_20171009_en.txt", self._tokenize_en
        )
        corpus_tgt = self._get_tokenization(
            "../data/gc_2010-2017_conglomerated_20171009_es.txt", self._tokenize_es
        )

        corpus = list(zip(corpus_src, corpus_tgt))
        lens = np.array([(len(obs_src), len(obs_tgt)) for obs_src, obs_tgt in corpus])
        max_len = np.max(lens, axis=1)
        corpus = [elem for elem, num in zip(corpus, max_len) if num <= self.MAX_LEN]

        self.corpus_pre = []
        for elem_src, elem_tgt in corpus:
            elem_src += [self.PADDING] * (self.MAX_LEN - len(elem_src) + 1) # +1 because we are adding <SOS> or <EOS> in target
            elem_tgt += [self.PADDING] * (self.MAX_LEN - len(elem_tgt))
            self.corpus_pre.append((elem_src, elem_tgt))

        # TODO: improve, placeholder to see if it works
        self.voc_en = self.SPECIAL_TOKENS + list(set(x for elem in (elem for elem, _ in self.corpus_pre) for x in elem if x not in self.SPECIAL_TOKENS))
        self.voc_en_map = {voc: i for i, voc in enumerate(self.voc_en)}

        self.voc_es = self.SPECIAL_TOKENS + list(set(x for elem in (elem for _, elem in self.corpus_pre) for x in elem if x not in self.SPECIAL_TOKENS))
        self.voc_es_map = {voc: i for i, voc in enumerate(self.voc_es)}
        
        self.sos = self._token_to_idx([self.START_OF_SENTENCE], False).unsqueeze(1)
        self.eos = self._token_to_idx([self.END_OF_SENTENCE], False).unsqueeze(1)


    def _get_tokenization(self, file_pth, tokenizer):
        with open(file_pth, "r", encoding="utf8") as f:
            text = f.read().split("\n")
            return [tokenizer(line) for line in text]

    def _tokenize_es(self, text):
        return [tok.text for tok in self.spacy_es.tokenizer(text)]

    def _tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def _token_to_idx(self, X, is_en=True):
        if is_en:
            voc_map = self.voc_en_map
        else:
            voc_map = self.voc_es_map
        embedding = np.zeros((len(X)))
        for i, word in enumerate(X):
            embedding[i] = voc_map[word]
        return torch.tensor(embedding, dtype=torch.long)

    def _idx_to_token(self, X, is_en=False):
        voc = self.voc_es
        if is_en:
            voc = self.voc_en
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
        elem_src, elem_tgt = self.corpus_pre[idx]
        return self._token_to_idx(elem_src), self._token_to_idx(elem_tgt, False), elem_src, elem_tgt
