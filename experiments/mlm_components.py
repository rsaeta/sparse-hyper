import gzip
import os

import numpy as np
import torch

import string


class ByteTokenizer:

    def __init__(self):
        self.sep_token = 20e10
        self.min_token = 20e10
        self.max_token = 20e10
        self.mask_token = 20e10
        self.pad_token = 20e10
        self.max_len = -1
        self.train([])

    def train(self, datas, **kwargs):
        chrs = set()
        chrs = chrs.union(set(map(torch.tensor, map(ord, string.printable))))
        for d in datas:
            text, val, test = enwik8(d)
            chrs = chrs.union(text.unique())
        self.min_token = min(chrs).item()
        self.max_token = max(chrs).item()
        self.mask_token = self.max_token+1
        self.pad_token = self.mask_token+1
        self.sep_token = self.pad_token+1

    def get_vocab_size(self, with_added_tokens=True):
        return self.mask_token+3 if with_added_tokens else self.mask_token

    def token_to_id(self, token):
        if token == '[MASK]':
            return self.mask_token
        elif token == '[PAD]':
            return self.pad_token
        elif token == '[SEP]':
            return self.sep_token
        return self.encode(token).ids

    def enable_padding(self, *, length):
        self.max_len = length

    def encode(self, s):
        ids = list(map(ord, s))
        attentions = [1]*len(ids)
        if self.max_len > 0:
            num_pad = self.max_len - len(ids)
            ids += [self.pad_token] * num_pad
            attentions += [0] * num_pad
        encs = ByteEncoding(ids, attentions)
        return encs

    def __call__(self, *args, **kwargs):
        return self.encode(*args)


class ByteEncoding:

    def __init__(self, ids, masks):
        self.ids = ids
        self.attention_mask = masks


def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.
    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    file_stats = os.stat(path)
    size = file_stats.st_size
    n_train = int(size*.9)
    n_valid = n_test = int(size*.05)
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)
