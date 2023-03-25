import os
import json
import torch
from functools import partial
from transformer_models import GeneratingTransformer
from argparse import Namespace, ArgumentParser
import gzip
import numpy as np


cuda = torch.cuda.is_available()


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
    parser.add_argument('-A', '--attention-type', choices=['dense', 'sparse', 'fixed', 'sparse2d', 'native', 'dilated'],
                        dest='attention_type', default='dense', type=str)
    parser.add_argument('-L', '--clipping-value', type=float,
                        dest='clipping_value', default=1.0)
    parser.add_argument('-Y', '--save-every', default=1,
                        type=int, dest='save_every')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='./saved_models')
    parser.add_argument('--watch-model', dest='watch_model', action='store_true')
    parser.add_argument('--plot-attention', dest='plot_attention', action='store_true')
    parser.add_argument('--load-model', dest='load_model',
                        default=None, type=str)
    parser.add_argument('--interact', default=False, action='store_true')
    options = parser.parse_args()
    print(options)
    return options


def get_model(args: Namespace, mask: bool = False) -> GeneratingTransformer:
    model = GeneratingTransformer(
        args.depth,
        args.context,
        args.embedding,
        256,
        k=args.num_indices,
        heads=args.n_heads,
        nadditional=args.nadditional,
        gadditional=args.gadditional,
        attention_type=args.attention_type,
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
        return (i+1)/args.lr_warmup
    else:
        return 1 - i/args.num_batches


def learners(model, args):
    optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  partial(lr, args))
    if args.load_model is not None:
        optimizername = args.load_model.replace('model.pt', 'optimizer.pt')
        schedulername = args.load_model.replace('model.pt', 'scheduler.pt')
        if os.path.isfile(optimizername):
            state_dict = torch.load(optimizername, map_location=torch.device('cuda') \
                                    if cuda else torch.device('cpu'))
            optimizer.load_state_dict(state_dict)
        if os.path.isfile(schedulername):
            state_dict = torch.load(schedulername, map_location=torch.device('cuda') \
                                    if cuda else torch.device('cpu'))
            scheduler.load_state_dict(state_dict)

    return optimizer, scheduler


def setup(args: Namespace):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)
