import torch
import toolz as tlz
import unicodedata
import re
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_data(sentences: np.ndarray):

    def stringify(x): return str(x)

    def unicode2ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def normalize_string(s):
        s = unicode2ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    return tlz.thread_last(sentences,
                           (map, stringify),
                           (map, normalize_string)
                           )

