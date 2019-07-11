import os
import time
import rootpath as rp
from torch.nn.utils.rnn import pack_sequence
from src.models.embedder import Embedder
from src.models.seq2seq import Encoder, Decoder
from src.data_helpers.data_loader import MoviesDialoguesDataset

rootpath = rp.detect()
cornell_data_set = os.path.join(rootpath, 'data', 'cornell movie-dialogs corpus')

movies_dataset = MoviesDialoguesDataset(cornell_corpus_path=cornell_data_set, movie_name='star wars')

sentences = movies_dataset[0:2]

ts = time.time()
embedder = Embedder()
print(time.time() - ts)

ts = time.time()
embeddings = embedder.embed(sentences['input'])
print(time.time() - ts)

# embeddings = pack_sequence(embeddings, enforce_sorted=False)
encoder = Encoder(input_size=len(embedder), hidden_size=256)
hidden = encoder.init_hidden()
out, hidden = encoder(embeddings, hidden)

decoder = Decoder(hidden_size=256, output_size=10, max_length=30)
dec_hid = decoder.init_hidden(len(sentences))
decoder_hidden = dec_hid
encoder_outputs = out
x = hidden
output, hidden = decoder(dec_hid, out, hidden)