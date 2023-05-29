import argparse
import tokenizers
import os
from omegaconf import OmegaConf
import json
import re
from pathlib import Path

import torch
import wandb

from lib.models import GeneratingTransformer, ClassificationTransformer

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


def lr(lr_warmup, num_batches, min_lr, i):
    if i < lr_warmup:
        next_lr = (i+1)/lr_warmup
    else:
        next_lr = 1 - i/num_batches
    return max(next_lr, min_lr)


def learners(model, cfg: RunConfig, load=True):
    optimizer = torch.optim.AdamW(lr=cfg.experiment.optim.lr,
                                  betas=tuple(cfg.experiment.optim.betas),
                                  eps=cfg.experiment.optim.eps,
                                  weight_decay=cfg.experiment.optim.weight_decay,
                                  params=model.parameters())
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


def get_tokenizer(cfg: RunConfig) -> tokenizers.Tokenizer:
    tok_cfg = cfg.experiment.data.tokenizer
    tok_type = tok_cfg.type
    if tok_type == 'wordpiece':
        tokenizer_cls = tokenizers.BertWordPieceTokenizer
    else:
        raise NotImplementedError(f'Tokenizer {cfg.experiment.data.tokenizer.type} not yet implemented')

    if tok_cfg.load_file is not None:
        tok = tokenizer_cls.from_file(cfg.experiment.data.tokenizer.load_file)
    elif tok_cfg.train_file is not None:
        tok = tokenizer_cls()
        tok.train(tok_cfg.train_file, vocab_size=tok_cfg.vocab_size)
    tok.enable_padding(length=cfg.experiment.context_size)
    return tok


def get_model(cfg: RunConfig):
    model_cls = ClassificationTransformer \
        if cfg.experiment.data.output_type == 'classification' \
        else GeneratingTransformer
    model = model_cls(cfg.model)
    return model.to(device)


def get_criterion(cfg: RunConfig):
    if cfg.experiment.data.output_type == 'classification':
        if cfg.experiment.data.num_classes == 2:
            return torch.nn.functional.binary_cross_entropy_with_logits
        return torch.nn.functional.cross_entropy
    return torch.nn.functional.mse_loss


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
        config=dict(cfg),
    )


def post_process_cfg(cfg: OmegaConf) -> RunConfig:
    """
    Because of shenanigans with not supporting well list interpolation, there's some interesting
    things I had to do. See: https://github.com/facebookresearch/hydra/issues/1939#issuecomment-1035395006
    """
    OmegaConf.resolve(cfg)  # resolve interpolation
    cfg = OmegaConf.structured(RunConfig(**cfg))  # type check stuff
    del cfg.model['_t_block_dict']  # delete hackery
    return cfg
