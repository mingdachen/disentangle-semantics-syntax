import torch
import model_utils
import torch.nn as nn


class encoder_base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, log,
                 *args, **kwargs):
        super(encoder_base, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
            log.info(
                "{} initialized with pretrained word embedding".format(
                    type(self)))


class word_avg(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, log,
                 *args, **kwargs):
        super(word_avg, self).__init__(vocab_size, embed_dim, embed_init, log)

    def forward(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        sum_vecs = (input_vecs * mask.unsqueeze(-1)).sum(1)
        avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
        return input_vecs, avg_vecs


class bilstm(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, hidden_size, log,
                 *args, **kwargs):
        super(bilstm, self).__init__(vocab_size, embed_dim, embed_init, log)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, mask, temp=None):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return input_vecs, sent_vec
