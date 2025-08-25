import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from .base_rnn import BaseRNN
from .attention import Attention

from Model import GIN
from Match import GraphMatcher
import utils.utils as utils

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")


def logical_or(x, y):
    return (x + y).clamp_(0, 1)


def logical_not(x):
    return x == 0


class Decoder(BaseRNN):
    """Decoder RNN module
    vocab_size: Number of tokens in the input language
    max_len: maximum length of an input
    word_vec_dim: embedding dimension used to embed tokens
    hidden_size: Hidden size of one RNN unit.
    n_layers: Number of layers in the RNN
    start_id: ID of the token representing start
    end_id: ID of the token representing end
    rnn_cell: Type of RNN cell being used by the model. Default=lstm
    dropout: Proportion of nodes turned off randomly while training
    use_attention: If the model should attend over the input tensor.
    """

    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size,
                 n_layers, start_id=1, end_id=2, rnn_cell='lstm',
                 bidirectional=False, input_dropout_p=0,
                 dropout_p=0, use_attention=False):
        super(Decoder, self).__init__(vocab_size, max_len, hidden_size,
                                      input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.max_length = max_len
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.word_vec_dim = word_vec_dim
        self.bidirectional_encoder = bidirectional
        if bidirectional:
            self.hidden_size *= 2
        self.use_attention = use_attention
        self.start_id = start_id
        self.end_id = end_id

        self.embedding = nn.Embedding(self.output_size, self.word_vec_dim)
        self.rnn = self.rnn_cell(self.word_vec_dim, self.hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = self.out_linear(output.contiguous().view(-1, self.hidden_size))
        predicted_softmax = F.log_softmax(output.view(batch_size, output_size, -1), 2)
        return predicted_softmax, hidden, attn

    def forward(self, y, encoder_outputs, encoder_hidden):
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn = self.forward_step(y, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_sample(self, encoder_outputs, encoder_hidden, reinforce_sample=False):
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
        else:
            batch_size = encoder_hidden.size(1)
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.start_id))
        decoder_input = decoder_input.to(encoder_hidden[0].device)

        output_symbols = [decoder_input]
        output_logprobs = [torch.zeros(batch_size).to(decoder_input.device)]
        done = torch.ByteTensor(batch_size).fill_(0).to(decoder_input.device)

        def decode(i, output, reinforce_sample=reinforce_sample):
            nonlocal done
            if reinforce_sample:
                dist = torch.distributions.Categorical(
                    probs=torch.exp(output.view(batch_size, -1)))  # better initialize with logits
                symbols = dist.sample().unsqueeze(1)
            else:
                symbols = output.topk(1)[1].view(batch_size, -1)
            symbol_logprobs = output[:, 0, :][torch.arange(batch_size), symbols[:, 0]]
            not_done = logical_not(done)
            output_logprobs.append(not_done.float() * symbol_logprobs)
            output_symbols.append(symbols)
            done = logical_or(done, symbols[:, 0] == self.end_id)
            return symbols

        for i in range(self.max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            decoder_input = decode(i, decoder_output)

        return output_symbols, output_logprobs

    def forward_sample_ns_vqa(self, encoder_outputs, encoder_hidden, reinforce_sample=False):
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
            use_cuda = encoder_hidden[0].is_cuda
        else:
            batch_size = encoder_hidden.size(1)
            use_cuda = encoder_hidden.is_cuda
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.start_id))
        if use_cuda:
            decoder_input = decoder_input.cuda()

        output_logprobs = []
        output_symbols = []
        output_lengths = np.array([self.max_length] * batch_size)

        def decode(i, output, reinforce_sample=reinforce_sample):
            output_logprobs.append(output.squeeze())
            if reinforce_sample:
                dist = torch.distributions.Categorical(
                    probs=torch.exp(output.view(batch_size, -1)))  # better initialize with logits
                symbols = dist.sample().unsqueeze(1)
            else:
                symbols = output.topk(1)[1].view(batch_size, -1)
            output_symbols.append(symbols.squeeze())

            eos_batches = symbols.data.eq(self.end_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((output_lengths > i) & eos_batches) != 0
                output_lengths[update_idx] = len(output_symbols)

            return symbols

        for i in range(self.max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            decoder_input = decode(i, decoder_output)

        return output_symbols, output_logprobs

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class Encoder(BaseRNN):
    """Encoder RNN module
    vocab_size: Number of tokens in the input language
    max_len: maximum length of an input
    word_vec_dim: embedding dimension used to embed tokens
    hidden_size: Hidden size of one RNN unit.
    n_layers: Number of layers in the RNN
    input_dropout: Proportion of nodes turned off randomly in the input
    dropout: Proportion of nodes turned off randomly while training
    bidirectional: Whether a bidirectional RNN should be trained for the given task
    rnn_cell: Type of RNN cell being used by the model. Default=lstm
    variable_lengths: If variable length inputs should be allowed
    word2vec: Whether word2vec embedding should be used over a standard embedding layer
    use_attention: If the model should attend over the input tensor.
    """
    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size, n_layers,
                 input_dropout_p=0, dropout_p=0, bidirectional=False, rnn_cell='lstm',
                 variable_lengths=False, word2vec=None, fix_embedding=False, gembd_vec_dim=0):
        super(Encoder, self).__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        if word2vec is not None:
            assert word2vec.size(0) == vocab_size
            self.word_vec_dim = word2vec.size(1)
            self.embedding = nn.Embedding(vocab_size, self.word_vec_dim)
            self.embedding.weight = nn.Parameter(word2vec)
        else:
            self.word_vec_dim = word_vec_dim
            self.embedding = nn.Embedding(vocab_size, word_vec_dim)
        if fix_embedding:
            self.embedding.weight.requires_grad = False

        self.gembd_vec_dim = gembd_vec_dim
        rnn_input_dim = self.word_vec_dim + self.gembd_vec_dim
        self.rnn = self.rnn_cell( rnn_input_dim, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bool(bidirectional), dropout=dropout_p)

    def forward(self, x, g_embd, input_lengths=None):
        """
        :param input_lengths:
        :param x question token sequence  (bsz, max(q_seq_len)) e.g. (64, 41)
        :param g_embd averaged Gs, Gt embedding (bsz, g_embd_vec) e.g. (64, 96)
        To do: add input, output dimensions to docstring
        """
        if self.gembd_vec_dim > 0:
            g_embedding = g_embd.unsqueeze(1).repeat(1, x.size(1), 1)
            word_embedding = self.embedding(x)
            embedded = torch.cat((word_embedding, g_embedding), dim=2)
        else:
            # Baseline model
            embedded = self.embedding(x)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def forward2(self, input_var, input_lengths=None):
        """
                :param input_var question token sequence
                To do: add input, output dimensions to docstring
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class Seq2seqGNN(nn.Module):
    """Seq2seqGNN model module
    Encoder: Object representing the encoder model
    Decoder: Object representing the decoder model
    GNN: Object representing the Graphical Neural Network
    """

    def __init__(self, encoder, decoder, gnn=None):
        super(Seq2seqGNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gnn = gnn

    def _gnn_forward(self, g_data):
        """
        Two feeding mechanisms:
        1. End-to-end: GNN (GraphMatcher, GIN(gs), GIN(gt)) are learnt
        2. Pre-trained: pretrained embedding vector (dim=g_embd_dim) is directly used.

        :param g_data: Graph data
        :return: g_embd, the joint Gs, Gt embedding used for augmenting seq2seq
        """
        is_end2end_flow = self.gnn and type(g_data) == tuple and (type(g_data[0]).__name__ == 'Batch' or type(g_data[0]).__name__ == 'DataBatch')
        Phi = None
        if is_end2end_flow:
            BATCH_S, BATCH_T = g_data
            x_s, edge_index_s, edge_attr_s, batch_s = BATCH_S.x, BATCH_S.edge_index, BATCH_S.edge_attr, torch.ones_like(BATCH_S.x).view(1, -1)[0, 0].view(-1,).long()
            x_t, edge_index_t, edge_attr_t, batch_t = BATCH_T.x, BATCH_T.edge_index, BATCH_T.edge_attr, torch.ones_like(BATCH_T.x).view(1, -1)[0, 0].view(-1,).long()
            g_embd, Phi = self.gnn(x_s, edge_index_s, edge_attr_s, None,
                               x_t, edge_index_t, edge_attr_t, None)
        else:
            g_embd = g_data  # when pre-trained g_embd is directed used as embd vec
        # Graphmatcher only returning S
        return g_embd, Phi

    def forward(self, x, y, g_data, input_lengths=None):
        """
        Notes:
        1. Feed the g_data -> GNN -> g_embeds
        2. Use g_embeds [1] as usual. Keep g_embeds dim same for now (i.e. 96)

        :param x: questions
        :param y: programs (ground truths)
        :param g_data: graph data (Gs, Gt or Gu: variants)
        :param input_lengths: len of x, used for RNN (un)packing
        :return: decoder outputs
        """
        g_embd, Phi = self._gnn_forward(g_data)
        # g_embd should be the embeddings of only the corresponding vectors not all the nodes
        # in Gs, Gt.
        #  : Keeping encoder, decoder the same)
        g_embd = torch.mean(g_embd, dim=1)
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs

    def sample_output(self, x, g_data, input_lengths=None):
        g_embd, Phi = self._gnn_forward(g_data)
        g_embd = torch.mean(g_embd, dim=1)
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        # import pdb
        # pdb.set_trace()
        return torch.stack(output_symbols).transpose(0, 1)

    def reinforce_forward(self, x, g_data, input_lengths=None):
        g_embd, Phi = self._gnn_forward(g_data)
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        self.output_symbols, self.output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden,
                                                                                reinforce_sample=True)
        return torch.stack(self.output_symbols).transpose(0, 1)

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        # the output_symbols, output_logprobs were calculated in the reinforce_forward step.
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1:  # one-dim index values
                logprob_pred_symbol = torch.index_select(self.output_logprobs[i], 1, symbol)
                logprob_pred = torch.diag(logprob_pred_symbol).sum()
                probs_i = torch.exp(self.output_logprobs[i])
                plogp = self.output_logprobs[i] * probs_i
                entropy_offset = entropy_factor * plogp.sum()

                loss = - logprob_pred * reward + entropy_offset
                print(f"i = {i}: loss = - {logprob_pred} * {reward} + {entropy_offset} = {loss}")
            else:
                loss = - self.output_logprobs[i] * reward
            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)

    def __repr__(self):
        return ('{}(\n'
                '   encoder={}\n'
                '   decoder={}\n'
                '   gnn={}\n)').format(self.__class__.__name__,
                                       self.encoder,
                                       self.decoder,
                                       self.gnn)


def get_vocab(opt):
    if opt.dataset == 'clevr':
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    vocab = utils.load_vocab(vocab_json)
    return vocab


def create_seq2seq_net(input_vocab_size, output_vocab_size, hidden_size,
                       word_vec_dim, n_layers, bidirectional, variable_lengths,
                       use_attention, encoder_max_len, decoder_max_len, start_id,
                       end_id, word2vec_path=None, fix_embedding=False, gembd_vec_dim=0):
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len,
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding, gembd_vec_dim=gembd_vec_dim)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    return Seq2seq(encoder, decoder)


def create_seq2seq_gnn_net(input_vocab_size, output_vocab_size, hidden_size,
                       word_vec_dim, n_layers, bidirectional, variable_lengths,
                       use_attention, encoder_max_len, decoder_max_len, start_id,
                       end_id, gnn, gembd_vec_dim, is_padding_pos, word2vec_path=None, fix_embedding=False, **kwargs):
    """ Creates a Seq2seqGNN model for e2e training for Gs, Gt embedding """
    """
        input_vocab_size: number of distinct tokens in the input language (question space)
        output_vocab_size: number of distinct tokens in teh output language (answer space)
        hidden_size: the hidden size used by encoder and decoder models in their RNN cells
        word_vec_dim: number of dimensions used to embed words/tokens into vectors
        n_layers: number of RNN layers used by the encoder and decoder models 
        bidirectional: whether a bidirectional RNN is trained for encoder and decoder
        variable_lengths: whether we allow variable length inputs for the encoder and decoder models
        use_attention: If the model should attend over the input tensor.
        encoder_max_len: Maximum length of the input to the encoder.
        decoder_max_len: Maximum length of the input to the decoder.
        start_id: Token number representing the start token
        end_id: Token number representing the end token
        GNN: Class that will be used to instantiate the GNN model
        gembd_vec_dim: The number of dimensions used by the GNN while embedding vectors
        is_padding_pos: whether padding was used while tokenizing the input
    """
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len,
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding, gembd_vec_dim=gembd_vec_dim)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    logging.info(f"Instantiating GNN type: {gnn}")
    in_channels = gembd_vec_dim+3 if is_padding_pos else gembd_vec_dim      # 99 | 96
    psi_1 = GIN(in_channels=in_channels, out_channels=48, num_layers=2)
    gnn = GraphMatcher(psi_1, gembd_vec_dim=gembd_vec_dim, aggregation='cat')

    return Seq2seqGNN(encoder, decoder, gnn)