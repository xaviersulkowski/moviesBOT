import torch

from typing import Union, List
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings


class Embedder:
    def __init__(self):
        self.embedder = DocumentRNNEmbeddings([FlairEmbeddings('news-forward-fast')])
        self.embedding_length = self.__len__()

    def __len__(self):
        return self.embedder.embedding_length

    def __call__(self, sentences: Union[str, List[str]]):
        return self.embed(sentences)

    def embed(self, sentences: Union[str, List[str]]):
        if type(sentences) not in [str, list]:
            raise TypeError(f'Expected str or List[str] input got {type(sentences)}')

        if type(sentences) == str:
            sentences = [sentences]

        sentences = [Sentence(sentence) for sentence in sentences]

        self.embedder.embed(sentences)
        embeddings = [sentence.get_embedding().unsqueeze(0) for sentence in sentences]
        embeddings = torch.cat(embeddings, 0)
        return embeddings.unsqueeze(0)
