from typing import Union, List
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings


class Embedder:
    def __init__(self):
        self.embedder = FlairEmbeddings('news-forward-fast')
        self.embedding_length = self.__len__()

    def __len__(self):
        return self.embedder.embedding_length

    def __call__(self, sentences: Union[str, List[str]]):
        return self.embed(sentences)

    def embed(self, sentences: Union[str, List[str]]):
        if type(sentences) == str:
            sentences = Sentence(sentences)
        elif type(sentences) == list:
            sentences = [Sentence(sentence) for sentence in sentences]
        else:
            raise TypeError(f'Expected str or List[str] input got {type(sentences)}')

        embeddings = self.embedder.embed(sentences)
        return embeddings
