import torch
import model_utils

import torch.nn as nn
import torch.nn.functional as F


class bag_of_words(nn.Module):
    def __init__(self, ysize, zsize, mlp_layer, hidden_size,
                 vocab_size, dropout, *args, **kwargs):
        super(bag_of_words, self).__init__()
        self.hid2vocab = model_utils.get_mlp(
            ysize + zsize,
            hidden_size,
            vocab_size,
            mlp_layer,
            dropout)

    def forward(self, yvecs, zvecs, tgts, tgts_mask):
        input_vecs = torch.cat([yvecs, zvecs], -1)
        logits = F.log_softmax(self.hid2vocab(input_vecs), -1)
        return -(torch.sum(logits * tgts, 1) / tgts.sum(1)).mean()


class lstm(nn.Module):
    def __init__(self, ysize, zsize, vocab_size, mlp_hidden_size,
                 mlp_layer, hidden_size, dropout,
                 *args, **kwargs):
        super(lstm, self).__init__()
        self.cell = nn.LSTM(
            zsize, hidden_size,
            bidirectional=False, batch_first=True)
        self.hid2vocab = model_utils.get_mlp(
            hidden_size + ysize,
            hidden_size,
            vocab_size,
            mlp_layer,
            dropout)

    def forward(self, yvecs, zvecs, tgts, tgts_mask,
                *args, **kwargs):
        bs, sl = tgts_mask.size()
        ex_input_vecs = zvecs.unsqueeze(1).expand(-1, sl, -1)
        ex_output_vecs = yvecs.unsqueeze(1).expand(-1, sl, -1)

        ori_output_seq, _ = model_utils.get_rnn_vecs(
            ex_input_vecs, tgts_mask, self.cell)
        output_seq = torch.cat([ori_output_seq, ex_output_vecs], -1)
        # batch size x seq len x vocab size
        pred = self.hid2vocab(output_seq)[:, :-1, :]

        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduce=False)

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean()
