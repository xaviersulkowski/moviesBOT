import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence.
        For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next
        input word.

        :param input_size: the number of features in the hidden state h -> embedding size
        :param hidden_size: the number of features in the hidden state
        """
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size,  batch_first=True)

    def init_hidden(self):
        """
        Initialize random hidden state

        :return: random hidden state
        """
        return torch.randn(1, 1, self.hidden_size, device=DEVICE)

    def forward(self, x, hidden):
        """
        Forward move.

        :param x:
        :param hidden: hidden layer from previous step
        :return: output and hidden state of gru
        """
        output, hidden = self.rnn(x, hidden)
        return output, hidden


# class Decoder(nn.Module):
#     def __init__(self, hidden_size: int, output_size: int, max_length: int):
#         """
#         The decoder is another RNN that takes the encoder output vector(s) and outputs a sequence of words to create
#         the answer.
#
#         :param hidden_size: the number of features in the hidden state
#         :param output_size: number of uniques words in dataset
#         """
#         super(Decoder, self).__init__()
#
#         self.max_length = max_length
#         self.hidden_size = hidden_size
#
#         self.attn = nn.Linear(hidden_size, max_length)
#         self.rnn = nn.GRU(hidden_size, output_size, batch_first=True)
#         self.linear_out = nn.Linear(output_size, max_length)
#
#     def init_hidden(self, n_sentences):
#         """
#         Initialize random hidden state
#
#         :return: random hidden state
#         """
#         return torch.randn(1, n_sentences, self.hidden_size, device=DEVICE)
#
#     def forward(self, decoder_hidden, encoder_outputs, x):
#         """
#         Forward move.
#         """
#
#         weights = []
#         for i in range(len(encoder_outputs)):
#             weights.append(self.attn(torch.cat((decoder_hidden[:, i, :], encoder_outputs[:, i, :]), dim=0)))
#
#         normalized_weights = F.softmax(torch.cat(weights, 1), 1)
#
#         attn_applied = torch.bmm(normalized_weights.unsqueeze(-1),
#                                  encoder_outputs.view(-1, 1, self.hidden_size))
#
#         input_rnn = torch.cat((attn_applied[0], x[0]), dim=1)
#
#         output, hidden = self.rnn(input_rnn.unsqueeze(0), decoder_hidden)
#
#         output = self.linear_out(output[0])
#
#         return output, hidden
