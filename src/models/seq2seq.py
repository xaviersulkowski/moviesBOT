import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1,
                 dropout: float = 0, bidirectional: bool = False):
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
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        dropout = 0 if n_layers == 1 else dropout
        # self.dropout = SpatialDropout(dropout)
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, dropout=dropout, batch_first=False)

    def forward(self, x: torch.nn.utils.rnn.PackedSequence, hidden=None):
        """
        Forward move.

        :param x:
        :param hidden: hidden layer from previous step
        :return: output and hidden state of gru
        """

        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x, _ = pad_packed_sequence(x, batch_first=False)

        output, hidden = self.rnn(x, hidden)

        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        general attention
        :param hidden_size: int
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.weights = nn.Linear(hidden_size, hidden_size, bias=False)

    def _score(self, hidden, encoder_outputs):
        x = self.weights(encoder_outputs)
        return torch.sum(hidden * x, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self._score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, -1).unsqueeze(1)


class AttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.8):
        super(AttentionDecoder, self).__init__()

        # inputs
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = 0 if n_layers == 1 else dropout

        # layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=False)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attention = Attention(hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):

        embeddings = self.embedding_dropout(input_step)
        rnn_output, hidden = self.rnn(embeddings, last_hidden)
        attn_weights = self.attention(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden
