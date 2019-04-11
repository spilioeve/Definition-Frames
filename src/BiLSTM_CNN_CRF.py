import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from itertools import repeat
import torch.nn.utils.rnn as rnn_utils
from Frame_Generator.BiLSTM_CNN import BiLSTM_CNN


START_TAG = "<START>"
STOP_TAG = "<STOP>"

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def argmaxs(mtx):
    vals,idxs = torch.max(mtx, 1)
    return vals,idxs.data.tolist()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exps(mtx):
    mtx.size()[0]
    max_scores,idxs = argmaxs(mtx)
    max_scores_broadcast = max_scores.view(mtx.size()[0],-1).expand(mtx.size()[0],mtx.size()[0])
    return max_scores + \
			torch.log(torch.sum(torch.exp((mtx - max_scores_broadcast)),dim=1))




class BiLSTM_CNN_CRF(BiLSTM_CNN):
    def __init__(self, labels, vocab_size, pos_size, chunk_size, char_size, embedding_dim, char_embed_dim, hidden_dim,
                 number_layers, batch_Size, num_filters, kernel_size):
        super(BiLSTM_CNN_CRF, self).__init__(labels, vocab_size, pos_size, chunk_size, char_size, embedding_dim, char_embed_dim, hidden_dim,
                 number_layers, batch_Size, num_filters, kernel_size)

    # def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels,
    #              tag_space=0, embedd_word=None, embedd_char=None, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5), bigram=False, initializer=None):
    #     super(BiRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels,
    #                                              tag_space=tag_space, embedd_word=embedd_word, embedd_char=embedd_char,
    #                                              p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        #
        # out_dim = tag_space if tag_space else hidden_size * 2
        # self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)
        # self.dense_softmax = None
        # self.logsoftmax = None
        # self.nll_loss = None

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask).mean()

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target).float().sum()
        else:
            return preds, (torch.eq(preds, target).float() * mask).sum()