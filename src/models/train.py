import os
import time
import torch
from src.utils.path_utils import project_root
from torch.nn.utils.rnn import pad_sequence
from src.models.embedder import Embedder
from src.models.seq2seq import Encoder, AttentionDecoder
from src.utils.data_loader import MoviesDialoguesDataset

rootpath = project_root()
cornell_data_set = os.path.join(rootpath, 'data', 'cornell movie-dialogs corpus')

movies_dataset = MoviesDialoguesDataset(cornell_corpus_path=cornell_data_set, movie_name='star wars')
batch_size = 3


lexicon = movies_dataset.create_lexicon()
input = movies_dataset.questions[:batch_size].values
output = movies_dataset.answers[:batch_size].values

# define
embedder = Embedder()
encoder = Encoder(input_size=len(embedder), hidden_size=256, n_layers=1, bidirectional=False, dropout=1.0)
decoder = AttentionDecoder(hidden_size=256, output_size=len(lexicon), dropout=1.0, input_size=len(embedder))

ts = time.time()
inputs = embedder.embed(input)
print(time.time() - ts)

# encoder
padded_inputs = pad_sequence(inputs, batch_first=False)
# out.shape= [max_length x n_sentences x hidden_size]
# hidden.shape = [(n_layers * num_directions) x n_sentences x hidden_size]
encoder_output, encoder_hidden = encoder(padded_inputs)

# decoder
decoder_hidden = encoder_hidden[:decoder.n_layers]
# decoder_input = torch.LongTensor([[1. for _ in range(3)]])
decoder_input = torch.ones((1, batch_size, len(embedder)))
decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
