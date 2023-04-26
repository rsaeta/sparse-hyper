import os
import json

import tokenizers
import torch
try:
    import torch._dynamo
    supports_dyno = True
except:
    supports_dyno = False
from functools import partial

from transformer_models import GeneratingTransformer
from argparse import Namespace, ArgumentParser
import gzip
import numpy as np
from tokenizers import BertWordPieceTokenizer

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

from transformer_models import attention_types

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

"""
compile = torch.compile
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True
"""


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-N', '--num-batches',
                        dest='num_batches',
                        default=1_000_000,
                        type=int)
    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        default=16,
                        type=int)
    parser.add_argument('-l', '--learning-rate',
                        dest='learning_rate',
                        default=0.001,
                        type=float)
    parser.add_argument('-D', '--data', type=str)
    parser.add_argument('-E', '--embedding',
                        default=4, type=int)
    parser.add_argument('-H', '--n-heads',
                        dest='n_heads',
                        default=4,
                        type=int)
    parser.add_argument("-C", "--context",
                        help="Length of the sequences extracted from the corpus and the context used during inference.",
                        default=256, type=int)
    parser.add_argument("-d", "--depth",
                        help="Depth of the network (nr. of transformer blocks)",
                        default=12, type=int)
    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)
    parser.add_argument('-k', '--num-indices',
                        dest='num_indices',
                        help='Number of points in floating-point indices',
                        default=8, type=int)
    parser.add_argument('-m', '--lr-warmup',
                        dest='lr_warmup',
                        default=500, type=int)
    parser.add_argument('-S', '--lr-stepsize',
                        dest='lr_stepsize',
                        default=10, type=int)
    parser.add_argument('-n', '--neighbor_sample',
                        dest='nadditional', default=2, type=int)
    parser.add_argument('-g', '--global_sample',
                        dest='gadditional', default=2, type=int)
    parser.add_argument('-V', '--validation-every', default=200,
                        dest='validation_every', type=int)
    parser.add_argument('-A', '--attention-type', choices=get_args(attention_types),
                        dest='attention_type', default='dense', type=str)
    parser.add_argument('-L', '--clipping-value', type=float,
                        dest='clipping_value', default=1.0)
    parser.add_argument('-Y', '--save-every', default=1,
                        type=int, dest='save_every')
    parser.add_argument('--save-dir', dest='save_dir', type=str, required=False)
    parser.add_argument('--watch-model', dest='watch_model', action='store_true')
    parser.add_argument('--plot-attention', dest='plot_attention', action='store_true')
    parser.add_argument('--load-model', dest='load_model',
                        default=None, type=str)
    parser.add_argument('--interact', default=False, action='store_true')
    parser.add_argument('--constant-lr', action='store_true')
    parser.add_argument('--tokenizer', default='wordpiece',
                        type=str, choices=['wordpiece', 'byte'])
    parser.add_argument('--tokenizer-file', default=None, required=False,
                        type=str, dest='tokenizer_file')
    parser.add_argument('--vocab-size', dest='vocab_size', default=32768,
                        type=int)
    parser.add_argument('--log-every', dest='log_every',
                        type=int, default=1)
    parser.add_argument('--min-lr', dest='min_lr', type=float, default=5e-5)
    parser.add_argument('--micro-batch-size', dest='micro_batch_size',
                        type=int, default=None)
    parser.add_argument('--model-type', dest='model_type', default=None, type=str)
    parser.add_argument('--resume-run', dest='resume_run', default=None, type=str)
    parser.add_argument('--save-last', dest='save_last_only', action='save_true', type=bool, default=False)
    options = parser.parse_args()
    return options


def get_model(args: Namespace, vocab_size: int, mask: bool = False) -> GeneratingTransformer:
    if args.model_type is None:
        attentions = None
    elif args.model_type == 'dabirds':
        attentions = [*['bigbird']*10, *['smallbird']*6]
    else:
        attentions = None if args.model_type is None else [
            'dilated',
            'dilated',
            'dilated',
            'dense',
            'dense',
            'simple-sparse',
            'simple-sparse',
            'simple-sparse',
            'sparse',
            'sparse',
            'sparse',
            'dense',
        ]
    model = GeneratingTransformer(
        args.depth,
        args.context,
        args.embedding,
        vocab_size=vocab_size,
        k=args.num_indices,
        heads=args.n_heads,
        nadditional=args.nadditional,
        gadditional=args.gadditional,
        attention_type=args.attention_type,
        attentions=attentions,
        mask=mask,
    )
    if args.load_model is not None:
        state_dict = torch.load(args.load_model, map_location=torch.device('cuda') \
                                if cuda else torch.device('cpu'))
        model.load_state_dict(state_dict)
    if cuda:
        model = model.cuda()
    return model


def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.
    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    file_stats = os.stat(path)
    size = file_stats.st_size
    n_train = int(size*.9)
    n_valid = n_test = int(size*.05)
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)


def lr(args, i):
    if i < args.lr_warmup:
        next_lr = (i+1)/args.lr_warmup
    else:
        next_lr = 1 - i/args.num_batches
    return max(next_lr, args.min_lr)


def learners(model, args, load=True):
    optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  partial(lr, args))
    if args.constant_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda _: 1.)
        return optimizer, scheduler
    if args.load_model is not None and load:
        optimizername = args.load_model.replace('model.pt', 'optimizer.pt')
        schedulername = args.load_model.replace('model.pt', 'scheduler.pt')
        if os.path.isfile(optimizername):
            state_dict = torch.load(optimizername, map_location=torch.device('cuda') \
                                    if cuda else torch.device('cpu'))
            optimizer.load_state_dict(state_dict)
        if os.path.isfile(schedulername) and not args.constant_lr:
            state_dict = torch.load(schedulername, map_location=torch.device('cuda') \
                                    if cuda else torch.device('cpu'))
            scheduler.load_state_dict(state_dict)

    return optimizer, scheduler


class ByteEncoding:

    def __init__(self, ids, masks):
        self.ids = ids
        self.attention_mask = masks


class ByteTokenizer:

    def __init__(self):
        self.min_token = 20e10
        self.max_token = 20e10
        self.mask_token = 20e10
        self.pad_token = 20e10
        self.max_len = -1

    def train(self, datas, **kwargs):
        chrs = set()
        for d in datas:
            text, val, test = enwik8(d)
            chrs = chrs.union(text.unique())
        self.min_token = min(chrs).item()
        self.max_token = max(chrs).item()
        self.mask_token = self.max_token+1
        self.pad_token = self.mask_token+1

    def get_vocab_size(self, with_added_tokens=True):
        return self.mask_token+2 if with_added_tokens else self.mask_token

    def token_to_id(self, token):
        if token == '[MASK]':
            return self.mask_token
        elif token == '[PAD]':
            return self.pad_token
        return self.encode(token).ids

    def enable_padding(self, *, length):
        self.max_len = length

    def encode(self, s):
        ids = list(map(ord, s))
        attentions = [1]*len(ids)
        if self.max_len > 0:
            num_pad = self.max_len - len(ids)
            ids += [self.pad_token] * num_pad
            attentions += [0] * num_pad
        encs = ByteEncoding(ids, attentions)
        return encs

    def __call__(self, *args, **kwargs):
        return self.encode(*args)


def get_tokenizer(args: Namespace) -> tokenizers.Tokenizer:
    if args.tokenizer == 'wordpiece':
        tokenizer_cls = BertWordPieceTokenizer
    elif args.tokenizer == 'byte':
        tokenizer_cls = ByteTokenizer
    else:
        raise NotImplementedError(f'Tokenizer {args.tokenizer} not yet implemented')
    tokenizer_fname = os.path.join('tokenizers', f'{args.tokenizer}_{os.path.basename(args.data)}.txt')
    if os.path.exists(tokenizer_fname):
        tok = tokenizer_cls.from_file(tokenizer_fname)
    else:
        tok = tokenizer_cls()
        tok.train([args.data], vocab_size=args.vocab_size)
    tok.enable_padding(length=args.context)
    return tok


def setup(args: Namespace):
    save_dir = args.save_dir
    if save_dir is None:
        return
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)
