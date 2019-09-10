import re
import json
import torch
import unicodedata
import numpy as np
import toolz as tlz

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


def read_hyperparameters(path_to_json: str) -> dict:
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    return data
