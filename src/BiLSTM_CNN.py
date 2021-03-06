import torch
import torch.nn as nn
import torch.nn.functional as F



class BiLSTM_CNN(nn.Module):
    def __init__(self, labels, vocab_size, pos_size, chunk_size, char_size, embedding_dim, char_embed_dim, hidden_dim, number_layers, batch_Size, num_filters, kernel_size):
        super(BiLSTM_CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.labels = labels
        self.labels_size = len(labels)
        self.batch_Size= batch_Size
        self.number_layers= number_layers
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx= 0)
        self.chunk_embeds = nn.Embedding(chunk_size, 10, padding_idx= 0)
        self.pos_embeds = nn.Embedding(pos_size, 10, padding_idx= 0)
        self.char_embeds = nn.Embedding(char_size, char_embed_dim, padding_idx= 0)

        self.conv1d = nn.Conv1d(char_embed_dim, num_filters, kernel_size, padding=kernel_size - 1)
        self.lstm = nn.LSTM(embedding_dim + 21+ num_filters, hidden_dim // 2,
                            num_layers=number_layers, bidirectional=True, batch_first=True) ##W dropout=0.5

        self.hidden2tag = nn.Linear(hidden_dim, self.labels_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden1= torch.randn(2*self.number_layers, self.batch_Size, self.hidden_dim // 2)
        hidden2 = torch.randn(2*self.number_layers, self.batch_Size, self.hidden_dim // 2)
        return (hidden1, hidden2)

    def forward(self, sentence, sentence_char, sentences_length):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        self.hidden = self.init_hidden()
        word_embeds = self.word_embeds(sentence[:, :, 0])
        pos_embeds = self.pos_embeds(sentence[:, :, 1])
        chunk_embeds= self.chunk_embeds(sentence[:, :, 2])
        flag= sentence[:, :, 3]
        flag= flag.unsqueeze(0).view((sentence.shape[0], sentence.shape[1], 1))
        flag= flag.type(torch.FloatTensor)
        char_embeds = self.char_embeds(sentence_char)
        char_size = char_embeds.size()
        char = char_embeds.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        char, _ = self.conv1d(char).max(dim=2)
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)
        embedding = torch.cat((word_embeds, pos_embeds, chunk_embeds, char, flag), 2) ###Changed
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(embedding, sentences_length, batch_first=True)
        X, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.hidden2tag(X)
        tag_space = F.log_softmax(X, dim=1)
        tag_scores = tag_space.view(self.batch_Size, sentence.shape[1], self.labels_size)
        return tag_scores

    def loss(self, y_pred, y, sentences_length):
        #loss = nn.NLLLoss()
        y = y.view(-1)
        y_pred= y_pred.view(-1, self.labels_size)
        mask = (y > 0).float()
        nb_tokens = int(torch.sum(mask).item())
        #nb_tokens = int(torch.sum(mask).data[0])
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens
        return  ce_loss