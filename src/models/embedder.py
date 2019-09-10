import torch
import numpy as np
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings


class Embedder:
    def __init__(self):
        self.embedder = FlairEmbeddings('news-forward-fast')
        self.embedding_length = self.__len__()

    def __len__(self):
        return self.embedder.embedding_length

    def __call__(self, sentences: np.ndarray):
        return self.embed(sentences)

    def embed(self, sentences: np.ndarray):
        if not isinstance(sentences, np.ndarray):
            raise TypeError(f'Expected numpy ndarray input got {type(sentences)}')

        if sentences.ndim != 2:
            raise TypeError(f'Expected numpy ndarray with 2 dims, try to A.reshape(-1, 1) ')

        sentences = [Sentence(sentence[0]) for sentence in sentences]

        self.embedder.embed(sentences)

        embeddings = []
        for sentence in sentences:
            embeddings.append(torch.stack([token.embedding.cpu() for token in sentence]))

        return embeddings
