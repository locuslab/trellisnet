import unidecode
import torch
from collections import Counter
import observations
import os
import pickle
import sys
sys.path.append("../")
from model import *


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, string):
        self.dictionary = Dictionary()
        for c in string:
            self.dictionary.add_word(c)
        self.dictionary.prep_dict()


def char_tensor(corpus, string):
    tensor = torch.zeros(len(string)).long()
    for i in range(len(string)):
        tensor[i] = corpus.dictionary.char2idx[string[i]]
    return tensor.cuda()


def repackage_hidden4(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, seq_len, evaluation=False):
    """`source` has dimension (L, N)"""
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data = source[i:i + seq_len]
    if evaluation:
        data.requires_grad = False
    target = source[i + 1:i + 1 + seq_len]  # CAUTION: This is un-flattened!
    return data, target


def save(model, args):
    save_filename = args.name + ".pt"
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
