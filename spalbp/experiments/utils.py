import argparse
from argparse import Namespace
import tokenizers
import os
from omegaconf import OmegaConf
import json
import re
from pathlib import Path

import torch
import wandb

try:
    import torch._dynamo
    supports_dyno = True
except:
    supports_dyno = False
from functools import partial

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

from config import RunConfig


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


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
                        help="Depth of the network (nr. of models blocks)",
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
    parser.add_argument('--save-last', dest='save_last_only', action='store_true')
    parser.add_argument('--finetune-ds', dest='finetune_ds', default=None, type=str)
    parser.add_argument('--production', action='store_true', dest='production')
    parser.add_argument('--finetune-subset', dest='finetune_subset', default=-1, type=int)
    parser.add_argument('--finetune-epochs', dest='finetune_epochs', default=5, type=int)
    parser.add_argument('--pos-embedding', dest='pos_embedding', default='learned', type=str,
                        choices=get_args(pos_encodings))
    parser.add_argument('--rand-data', dest='rand_data', action='store_true')
    options = parser.parse_args()
    return options


def lr(lr_warmup, num_batches, min_lr, i):
    if i < lr_warmup:
        next_lr = (i+1)/lr_warmup
    else:
        next_lr = 1 - i/num_batches
    return max(next_lr, min_lr)


def learners(model, cfg: RunConfig, load=True):
    optimizer = torch.optim.AdamW(lr=cfg.experiment.optim.lr, params=model.parameters())
    part = partial(lr,
                   cfg.experiment.scheduler.warmup_steps,
                   cfg.experiment.training.num_batches,
                   cfg.experiment.scheduler.min_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  part)
    if cfg.experiment.optim.type == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda _: 1.)
        return optimizer, scheduler
    if cfg.experiment.load_dir is not None and load:
        optimizer_name = os.path.join(cfg.experiment.load_dir, 'optimizer.pt')
        schedule_name = os.path.join(cfg.experiment.load_dir, 'scheduler.pt')
        if os.path.isfile(optimizer_name):
            state_dict = torch.load(optimizer_name, map_location=torch.device('cuda')
                                    if cuda else torch.device('cpu'))
            optimizer.load_state_dict(state_dict)
        if os.path.isfile(schedule_name) and not  cfg.experiment.optim.type == 'constant':
            state_dict = torch.load(schedule_name, map_location=torch.device('cuda')
                                    if cuda else torch.device('cpu'))
            scheduler.load_state_dict(state_dict)

    return optimizer, scheduler


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


def setup(cfg: RunConfig):
    save_dir = cfg.experiment.save_dir
    if save_dir is None:
        return
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


def find_latest_model(path: str) -> str:
    """Finds the latest model saved in path."""
    files = os.listdir(path)
    max_checkpoint = -1
    for f in files:
        match = re.match(r'.*/?checkpoint_(\d+)_model.pt', f)
        if match is not None:
            if int(match[1]) > max_checkpoint:
                max_checkpoint = int(match[1])
    checkpoint = os.path.join(path, f'checkpoint_{max_checkpoint}_model.pt')
    if os.path.exists(checkpoint):
        return checkpoint
    if os.path.exists(os.path.join(path, 'model.pt')):
        return os.path.join(path, 'model.pt')
    return None


def get_resume_args(args):
    config_fname = os.path.join(args.resume_run, 'config.json')
    latest_model_checkpoint = find_latest_model(args.resume_run)
    new_args = argparse.Namespace()
    with open(config_fname) as f:
        def_args = json.load(f)
    new_args.__dict__.update(**def_args)
    if args.save_dir is None:  # remap savedir
        save_dir = Path(args.resume_run).absolute()
        match = re.match(r'(.*)(\d+)$', save_dir.parts[-1])
        if match is not None:
            next_i = int(match[2]) + 1
            next_path = f'{match[1]}{next_i}'
        else:
            next_i = 2
            next_path = f'{save_dir.parts[-1]}{next_i}'
        full_path = os.path.join(*save_dir.parts[:-1], next_path)
        new_args.__dict__.update(save_dir=full_path)
    if latest_model_checkpoint is not None:
        new_args.__dict__.update(load_model=latest_model_checkpoint)
    if args.interact:
        new_args.__dict__.update(interact=True)
    new_args.__dict__.update(save_last_only=args.save_last_only)
    new_args.__dict__.update(production=args.production)
    return new_args


def save_model(cfg: RunConfig, model, optimizer, scheduler, checkpoint_num):
    f_name = f'{cfg.experiment.save_dir}/' if cfg.experiment.save_last \
        else f'{cfg.experiment.save_dir}/checkpoint_{checkpoint_num}_'
    torch.save(model.state_dict(), f_name + 'model.pt')
    torch.save(optimizer.state_dict(), f_name + 'optimizer.pt')
    torch.save(scheduler.state_dict(), f_name + 'scheduler.pt')


def init_wandb(cfg: RunConfig):
    wandb.init(
        project=cfg.experiment.wandb_project,
        config=cfg
    )
