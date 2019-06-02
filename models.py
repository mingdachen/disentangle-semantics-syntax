import os
import torch
import model_utils
import encoders
import decoders

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from von_mises_fisher import VonMisesFisher
from decorators import auto_init_args, auto_init_pytorch
from torch.autograd import Variable

MAX_LEN = 32


class base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.margin = self.expe.config.m
        self.use_cuda = self.expe.config.use_cuda

        self.yencode = getattr(encoders, self.expe.config.yencoder_type)(
            embed_dim=embed_dim,
            embed_init=embed_init,
            hidden_size=self.expe.config.ensize,
            vocab_size=vocab_size,
            log=experiment.log)

        self.zencode = getattr(encoders, self.expe.config.zencoder_type)(
            embed_dim=embed_dim,
            embed_init=embed_init,
            hidden_size=self.expe.config.ensize,
            vocab_size=vocab_size,
            log=experiment.log)

        if "lstm" in self.expe.config.yencoder_type.lower():
            y_out_size = 2 * self.expe.config.ensize
        elif self.expe.config.yencoder_type.lower() == "word_avg":
            y_out_size = embed_dim

        if "lstm" in self.expe.config.zencoder_type.lower():
            z_out_size = 2 * self.expe.config.ensize
        elif self.expe.config.zencoder_type.lower() == "word_avg":
            z_out_size = embed_dim

        self.mean1 = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)

        self.logvar1 = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=1,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)

        self.mean2 = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        self.logvar2 = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        if self.expe.config.zencoder_type.lower() == "word_avg":
            assert self.expe.config.decoder_type.lower() == "bag_of_words"

        self.decode = getattr(decoders, self.expe.config.decoder_type)(
            ysize=self.expe.config.ysize,
            zsize=self.expe.config.zsize,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            hidden_size=self.expe.config.desize,
            dropout=self.expe.config.dp,
            vocab_size=vocab_size)

        self.pos_decode = model_utils.get_mlp(
            input_size=self.expe.config.zsize + embed_dim,
            hidden_size=self.expe.config.mhsize,
            n_layer=self.expe.config.mlplayer,
            output_size=MAX_LEN,
            dropout=self.expe.config.dp)

    def pos_loss(self, mask, vecs):
        batch_size, seq_len = mask.size()
        # batch size x seq len x MAX LEN
        logits = self.pos_decode(vecs)
        if MAX_LEN - seq_len:
            padded = torch.zeros(batch_size, MAX_LEN - seq_len)
            new_mask = 1 - torch.cat([mask, self.to_var(padded)], -1)
        else:
            new_mask = 1 - mask
        new_mask = new_mask.unsqueeze(1).expand_as(logits)
        logits.data.masked_fill_(new_mask.data.byte(), -float('inf'))
        loss = F.softmax(logits, -1)[:, np.arange(int(seq_len)),
                                     np.arange(int(seq_len))]
        loss = -(loss + self.eps).log() * mask

        loss = loss.sum(-1) / mask.sum(1)
        return loss.mean()

    def sample_gaussian(self, mean, logvar):
        sample = mean + torch.exp(0.5 * logvar) * \
            Variable(logvar.data.new(logvar.size()).normal_())
        return sample

    def to_var(self, inputs):
        if self.use_cuda:
            if isinstance(inputs, Variable):
                inputs = inputs.cuda()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs.cuda(), volatile=self.volatile)
        else:
            if isinstance(inputs, Variable):
                inputs = inputs.cpu()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs, volatile=self.volatile)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and
                inputs_.size else None for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        return opt

    def save(self, dev_avg, dev_perf, test_avg,
             test_perf, epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_perf": dev_perf,
            "test_perf": test_perf,
            "dev_avg": dev_avg,
            "test_avg": test_avg,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="best"):
        if checkpointed_state_dict is None:
            save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage,
                                    loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            if checkpoint.get("opt_state_dict"):
                self.opt.load_state_dict(checkpoint.get("opt_state_dict"))

                if self.use_cuda:
                    for state in self.opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
            self.expe.log.info("model loaded from {}".format(save_path))
            return checkpoint.get('epoch', 0), \
                checkpoint.get('iteration', 0), \
                checkpoint.get('dev_avg', 0), \
                checkpoint.get('test_avg', 0), \
                checkpoint.get('kl_temp', 0)
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded!")

    @property
    def volatile(self):
        return not self.training


class vgvae(base):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(vgvae, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)

    def sent2param(self, sent, mask):
        yembed, yvecs = self.yencode(sent, mask)
        zembed, zvecs = self.zencode(sent, mask)

        mean = self.mean1(yvecs)
        mean = mean / mean.norm(dim=-1, keepdim=True)
        logvar = self.logvar1(yvecs)
        var = F.softplus(logvar) + 1

        mean2 = self.mean2(zvecs)
        logvar2 = self.logvar2(zvecs)

        return zembed, mean, var, mean2, logvar2

    def forward(self, sent1, mask1, sent2, mask2, tgt1,
                tgt_mask1, tgt2, tgt_mask2,
                neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                gtemp, use_margin):
        self.train()
        sent1, mask1, sent2, mask2, tgt1, \
            tgt_mask1, tgt2, tgt_mask2, neg_sent1, \
            neg_mask1, ntgt1, ntgt_mask1, neg_sent2, \
            neg_mask2, ntgt2, ntgt_mask2 = \
            self.to_vars(sent1, mask1, sent2, mask2, tgt1,
                         tgt_mask1, tgt2, tgt_mask2,
                         neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                         neg_sent2, neg_mask2, ntgt2, ntgt_mask2)

        s1_vecs, sent1_mean, sent1_var, sent1_mean2, sent1_logvar2 = \
            self.sent2param(sent1, mask1)
        s2_vecs, sent2_mean, sent2_var, sent2_mean2, sent2_logvar2 = \
            self.sent2param(sent2, mask2)

        sent1_dist = VonMisesFisher(sent1_mean, sent1_var)
        sent2_dist = VonMisesFisher(sent2_mean, sent2_var)

        sent1_syntax = self.sample_gaussian(sent1_mean2, sent1_logvar2)
        sent2_syntax = self.sample_gaussian(sent2_mean2, sent2_logvar2)

        sent1_semantic = sent1_dist.rsample()
        sent2_semantic = sent2_dist.rsample()

        logloss1 = self.decode(
            sent1_semantic, sent1_syntax, tgt1, tgt_mask1)
        logloss2 = self.decode(
            sent2_semantic, sent2_syntax, tgt2, tgt_mask2)

        logloss3 = self.decode(
            sent2_semantic, sent1_syntax, tgt1, tgt_mask1)
        logloss4 = self.decode(
            sent1_semantic, sent2_syntax, tgt2, tgt_mask2)

        if self.expe.config.pratio:
            s1_vecs = torch.cat(
                [s1_vecs, sent1_syntax.unsqueeze(1).expand(-1, s1_vecs.size(1), -1)], -1)
            s2_vecs = torch.cat(
                [s2_vecs, sent2_syntax.unsqueeze(1).expand(-1, s2_vecs.size(1), -1)], -1)
            ploss1 = self.pos_loss(mask1, s1_vecs)
            ploss2 = self.pos_loss(mask2, s2_vecs)

        sent1_kl = model_utils.gauss_kl_div(
            sent1_mean2, sent1_logvar2,
            eps=self.eps).mean()
        sent2_kl = model_utils.gauss_kl_div(
            sent2_mean2, sent2_logvar2,
            eps=self.eps).mean()

        if use_margin:

            n1_vecs, nsent1_mean, nsent1_var, nsent1_mean2, nsent1_logvar2 = \
                self.sent2param(neg_sent1, neg_mask1)
            n2_vecs, nsent2_mean, nsent2_var, nsent2_mean2, nsent2_logvar2 = \
                self.sent2param(neg_sent2, neg_mask2)

            nsent1_dist = VonMisesFisher(nsent1_mean, nsent1_var)
            nsent2_dist = VonMisesFisher(nsent2_mean, nsent2_var)

            nsent1_syntax = self.sample_gaussian(nsent1_mean2, nsent1_logvar2)
            nsent2_syntax = self.sample_gaussian(nsent2_mean2, nsent2_logvar2)

            nsent1_semantic = nsent1_dist.rsample()
            nsent2_semantic = nsent2_dist.rsample()

            logloss5 = self.decode(
                nsent1_semantic, nsent1_syntax,
                ntgt1, ntgt_mask1)
            logloss6 = self.decode(
                nsent2_semantic, nsent2_syntax,
                ntgt2, ntgt_mask2)

            if self.expe.config.pratio:
                n1_vecs = torch.cat(
                    [n1_vecs, nsent1_syntax.unsqueeze(1).expand(-1, n1_vecs.size(1), -1)], -1)
                n2_vecs = torch.cat(
                    [n2_vecs, nsent2_syntax.unsqueeze(1).expand(-1, n2_vecs.size(1), -1)], -1)
                ploss3 = self.pos_loss(neg_mask1, n1_vecs)
                ploss4 = self.pos_loss(neg_mask2, n2_vecs)

            nsent1_kl = model_utils.gauss_kl_div(
                nsent1_mean2, nsent1_logvar2,
                eps=self.eps).mean()
            nsent2_kl = model_utils.gauss_kl_div(
                nsent2_mean2, nsent2_logvar2,
                eps=self.eps).mean()

            sent_cos_pos = F.cosine_similarity(sent1_mean, sent2_mean)

            sent1_cos_neg = F.cosine_similarity(sent1_mean, nsent1_mean)
            sent2_cos_neg = F.cosine_similarity(sent2_mean, nsent2_mean)

            dist = F.relu(self.margin - sent_cos_pos + sent1_cos_neg) + \
                F.relu(self.margin - sent_cos_pos + sent2_cos_neg)

            dist = dist.mean()

            vkl = sent1_dist.kl_div().mean() + sent2_dist.kl_div().mean() + \
                nsent2_dist.kl_div().mean() + nsent1_dist.kl_div().mean()

            gkl = sent1_kl + sent2_kl + nsent1_kl + nsent2_kl

            rec_logloss = logloss1 + logloss2 + logloss5 + logloss6

            para_logloss = logloss3 + logloss4

            if self.expe.config.pratio:
                ploss = ploss1 + ploss2 + ploss3 + ploss4
            else:
                ploss = torch.zeros_like(gkl)

            loss = self.expe.config.lratio * rec_logloss + \
                self.expe.config.plratio * para_logloss + \
                vtemp * vkl + gtemp * gkl + \
                self.expe.config.dratio * dist + \
                self.expe.config.pratio * ploss

        else:
            vkl = sent1_dist.kl_div().mean() + sent2_dist.kl_div().mean()

            gkl = sent1_kl + sent2_kl

            rec_logloss = logloss1 + logloss2

            para_logloss = logloss3 + logloss4

            if self.expe.config.pratio:
                ploss = ploss1 + ploss2
            else:
                ploss = torch.zeros_like(gkl)

            loss = self.expe.config.lratio * rec_logloss + \
                self.expe.config.plratio * para_logloss + \
                vtemp * vkl + gtemp * gkl + \
                self.expe.config.pratio * ploss

            dist = torch.zeros_like(sent1_kl)

        return loss, vkl, gkl, rec_logloss, para_logloss, ploss, dist

    def score(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)

        sent1_vec = self.mean1(self.yencode(sent1, mask1)[1])
        sent2_vec = self.mean1(self.yencode(sent2, mask2)[1])

        return model_utils.pariwise_cosine_similarity(
            sent1_vec, sent2_vec).data.cpu().numpy()

    def pred(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)

        sent1_vec = self.mean1(self.yencode(sent1, mask1)[1])
        sent2_vec = self.mean1(self.yencode(sent2, mask2)[1])
        sent_cos_pos = F.cosine_similarity(sent1_vec, sent2_vec)
        return sent_cos_pos.data.cpu().numpy()

    def predz(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)

        _, sent1_vecs = self.zencode(sent1, mask1)
        _, sent2_vecs = self.zencode(sent2, mask2)
        sent1_mean1 = self.mean2(sent1_vecs)
        sent2_mean1 = self.mean2(sent2_vecs)

        sent_cos_pos = F.cosine_similarity(sent1_mean1, sent2_mean1)
        return sent_cos_pos.data.cpu().numpy()
