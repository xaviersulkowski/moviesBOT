import os
import rootpath as rp
from src.data_helpers.data_loader import MoviesDialoguesDataset
from src.models.embedder import Embedder
import time

rootpath = rp.detect()
cornell_data_set = os.path.join(rootpath, 'data', 'cornell movie-dialogs corpus')
print(cornell_data_set)

movies_dataset = MoviesDialoguesDataset(cornell_corpus_path=cornell_data_set, movie_name='star wars')

sentences = movies_dataset[0]

ts = time.time()
embedder = Embedder()
print(time.time() - ts)

ts = time.time()
embedder.embed(sentences['input'])
print(time.time() - ts)
