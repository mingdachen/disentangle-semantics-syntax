import argparse

UNK_IDX = 0
UNK_WORD = "UUUNKKK"
EVAL_YEAR = "2017"


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='Paraphrase using PyTorch')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_file', type=str, default=None,
                            help='training file')
    data_group.add_argument('--eval_file', type=str, default=None,
                            help='evaluation file')
    data_group.add_argument('--vocab_file', type=str, default=None,
                            help='vocabulary file')
    data_group.add_argument('--embed_file', type=str, default=None,
                            help='pretrained embedding file')

    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('-m', '--margin',
                              dest='m',
                              type=float,
                              default=0.4,
                              help='margin for the training loss')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=1e-3,
                              help='learning rate')
    config_group.add_argument('-pratio', '--ploss_ratio',
                              dest='pratio',
                              type=float,
                              default=1.0,
                              help='ratio of position loss')
    config_group.add_argument('-lratio', '--logloss_ratio',
                              dest='lratio',
                              type=float,
                              default=1.0,
                              help='ratio of reconstruction log loss')
    config_group.add_argument('-dratio', '--disc_ratio',
                              dest='dratio',
                              type=float,
                              default=1.0,
                              help='ratio of discriminative loss')
    config_group.add_argument('-plratio', '--para_logloss_ratio',
                              dest='plratio',
                              type=float,
                              default=1.0,
                              help='ratio of paraphrase log loss')
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-4,
                              help='for avoiding numerical issues')
    config_group.add_argument('-edim', '--embed_dim',
                              dest='edim',
                              type=int, default=50,
                              help='size of embedding')
    config_group.add_argument('-dp', '--dropout',
                              dest='dp',
                              type=float, default=0.0,
                              help='dropout probability')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=None,
                              help='gradient clipping threshold')
    # recurrent neural network detail
    config_group.add_argument('-ensize', '--encoder_size',
                              dest='ensize',
                              type=int, default=50,
                              help='encoder hidden size')
    config_group.add_argument('-desize', '--decoder_size',
                              dest='desize',
                              type=int, default=50,
                              help='decoder hidden size')
    config_group.add_argument('--ysize',
                              dest='ysize',
                              type=int, default=50,
                              help='size of vMF')
    config_group.add_argument('--zsize',
                              dest='zsize',
                              type=int, default=50,
                              help='size of Gaussian')

    # feedforward neural network
    config_group.add_argument('-mhsize', '--mlp_hidden_size',
                              dest='mhsize',
                              type=int, default=100,
                              help='size of hidden size')
    config_group.add_argument('-mlplayer', '--mlp_n_layer',
                              dest='mlplayer',
                              type=int, default=1,
                              help='number of layer')
    config_group.add_argument('-zmlplayer', '--zmlp_n_layer',
                              dest='zmlplayer',
                              type=int, default=1,
                              help='number of layer')
    config_group.add_argument('-ymlplayer', '--ymlp_n_layer',
                              dest='ymlplayer',
                              type=int, default=1,
                              help='number of layer')

    # optimization
    config_group.add_argument('-mb', '--mega_batch',
                              dest='mb',
                              type=int, default=1,
                              help='size of mega batching')
    config_group.add_argument('-ps', '--p_scramble',
                              dest='ps',
                              type=float, default=0.,
                              help='probability of scrambling')
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')
    config_group.add_argument('-vmkl', '--max_vmf_kl_temp',
                              dest='vmkl', type=float, default=1e-3,
                              help='temperature of kl divergence')
    config_group.add_argument('-gmkl', '--max_gauss_kl_temp',
                              dest='gmkl', type=float, default=1e-4,
                              help='temperature of kl divergence')

    setup_group = parser.add_argument_group('train_setup')
    # train detail
    setup_group.add_argument('--save_dir', type=str, default=None,
                             help='model save path')
    basic_group.add_argument('--embed_type',
                             type=str, default="paragram",
                             choices=['paragram', 'glove'],
                             help='types of embedding: paragram, glove')
    basic_group.add_argument('--yencoder_type',
                             type=str, default="word_avg",
                             help='types of encoder for y variable')
    basic_group.add_argument('--zencoder_type',
                             type=str, default="word_avg",
                             help='types of encoder for z encoder')
    basic_group.add_argument('--decoder_type',
                             type=str, default="bag_of_words",
                             help='types of decoder')
    setup_group.add_argument('--n_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--opt', type=str, default='adam',
                             help='types of optimizer')
    setup_group.add_argument('--pre_train_emb', type="bool", default=False,
                             help='whether to use pretrain embedding')
    setup_group.add_argument('--vocab_size', type=int, default=50000,
                             help='size of vocabulary')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=10,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=100,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--summarize', type="bool", default=False,
                            help='whether to summarize training stats\
                            (default: False)')
    return parser
