import datasets
import tokenizers
import transformers
import os
from omegaconf import OmegaConf
import re
import random

import torch
from torch.utils.data import DataLoader, Dataset
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
        return (i+1)/lr_warmup
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
        if os.path.isfile(schedule_name) and not cfg.experiment.optim.type == 'constant':
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


def load_dir(path: str):
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    model = get_model(cfg)
    model.load_state_dict(torch.load(find_latest_model(path),
                                     map_location=torch.device('cuda')
                                     if cuda else torch.device('cpu')))
    return cfg, model


def get_model(cfg: RunConfig):
    model_cls = ClassificationTransformer \
        if cfg.experiment.data.output_type == 'classification' \
        else GeneratingTransformer
    model = model_cls(cfg.model)
    if cfg.experiment.load_dir is not None:
        model_file = find_latest_model(cfg.experiment.load_dir)
        if model_file is None:
            raise FileNotFoundError(f'No model found in {cfg.experiment.load_dir}')
        model.load_state_dict(torch.load(model_file,
                                         map_location=torch.device('cuda')
                                         if cuda else torch.device('cpu')))
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
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.experiment.random_seed)


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


def get_new_save_dir(cur_save_dir: str):
    base_name = os.path.basename(cur_save_dir.strip('/'))
    # Look for the name to end in a digit
    match = re.match(r'(.*)-(\d+)', base_name)
    if match is not None:
        new_name = match[1] + '-' + str(int(match[2]) + 1)
    else:
        new_name = base_name + '-2'
    to_ret = os.path.join(os.path.dirname(cur_save_dir.strip('/')), new_name)
    return to_ret


def post_process_cfg(cfg: OmegaConf) -> RunConfig:
    """
    Because of shenanigans with not supporting well list interpolation, there's some interesting
    things I had to do. See: https://github.com/facebookresearch/hydra/issues/1939#issuecomment-1035395006
    """
    if cfg.resume is not None:
        cfg_file = os.path.join(cfg.resume, 'config.yaml')
        loaded_cfg = OmegaConf.load(cfg_file)
        cfg.model = loaded_cfg.model
        cfg.experiment.load_dir = cfg.resume
        cfg.experiment.save_dir = get_new_save_dir(cfg.resume)
    OmegaConf.resolve(cfg)  # resolve interpolation
    cfg = OmegaConf.structured(RunConfig(**cfg))  # type check stuff
    if '_t_block_dict' in cfg.model:
        del cfg.model['_t_block_dict']  # delete hackery
    return cfg


def get_pretrained_model(cfg: RunConfig):
    if 'teacher_model' not in cfg.experiment:
        raise ValueError('No teacher model specified')
    if cfg.experiment.teacher_model == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, model
    raise NotImplementedError(f'Pretrained model {cfg.experiment.teacher_model} not yet implemented')


class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset, tokenizer: transformers.PreTrainedTokenizer,
                 context_size: int, mask_prob: float, pad_middle=True):
        self.ds = dataset
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = context_size
        self.context_size = context_size
        self.mask_prob = mask_prob
        self.pad_middle = pad_middle

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        line1 = self.ds[item]['text']
        line2 = self.ds[item + 1]['text']

        tokens1 = self.tokenizer.tokenize(line1)
        tokens2 = self.tokenizer.tokenize(line2)
        num_1tokens = min(self.context_size - 2, len(tokens1))
        all_tokens = [self.tokenizer.cls_token, *(tokens1[:num_1tokens]), self.tokenizer.sep_token]
        num_2tokens = min(self.context_size - len(all_tokens), len(tokens2))
        middle_pad_amount = self.context_size - len(all_tokens) - num_2tokens
        random_ids = random.choices(list(self.tokenizer.get_vocab().values()), k=middle_pad_amount)
        all_tokens.extend(self.tokenizer.convert_ids_to_tokens(random_ids))
        all_tokens.extend(tokens2[:num_2tokens])
        assert len(all_tokens) == self.context_size, f'len(all_tokens) = {len(all_tokens)}, ' \
                                                     f'context_size = {self.context_size}'
        attention_mask = torch.ones((len(all_tokens),))[None, :].long()
        rand_mask = torch.rand(len(all_tokens)) < self.mask_prob
        targets = torch.tensor(self.tokenizer.convert_tokens_to_ids(all_tokens)).long()
        input_ids = torch.masked_fill(targets, rand_mask, self.tokenizer.mask_token_id)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets,
            'mask': rand_mask,
        }
