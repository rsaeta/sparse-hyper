import json
import os
import argparse
import re
from typing import Optional
from pathlib import Path
import git
import tokenizers

from experiment_utils import (
    parse_args,
    get_model,
    learners,
    setup,
    enwik8,
    get_tokenizer,
)

import torch
import torch.nn.functional as F

import wandb

from experiments.experiment_utils import cuda
from plot_utils import attention_viz


def init_wandb(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    wandb.init(
        project='sparse-masked-transformer',
        config={
            'context': args.context,
            'lr': args.learning_rate,
            'embedding': args.embedding,
            'depth': args.depth,
            'k': args.num_indices,
            'attention': args.attention_type,
            'gitsha': git.Repo(dir_path, search_parent_directories=True).head.object.hexsha,
            'model_type': args.model_type,
        }
    )


def sample_batch(data, tokenizer, length, batch_size, min_length, mask_p=0.15):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model
    while also randomly masking a single entry in the sequence to the mask_token provided.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :param tokenizer: The tokenizer that is used to parse the texts
    :param min_length: the min length of sequences to train on variable-length documents
    :param mask_p: The probability of masking a token
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """
    mask_token = tokenizer.token_to_id('[MASK]')
    byte_len = 10 * length

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - byte_len)
    lens = torch.randint(size=(batch_size,), low=min_length, high=length)

    # Slice out the input sequences
    strs = [ttos(data[start:start + byte_len]) for start in starts]
    attention_masks = []
    encoded_ids = []
    for s, l in zip(strs, lens):
        encoded = tokenizer.encode(s[:int(l * 4)])
        encoded_ids.append(encoded.ids[0:length])
        attention_masks.append(encoded.attention_mask[0:length])
    seqs_inputs = torch.tensor(encoded_ids)
    attention_masks = torch.tensor(attention_masks).bool()
    mask = torch.logical_and((torch.rand(seqs_inputs.size()) < mask_p), attention_masks)
    targets = seqs_inputs.detach().clone()
    seqs_inputs.masked_fill_(mask, mask_token)
    if cuda:
        seqs_inputs = seqs_inputs.cuda()
    c = attention_masks.size(-1)
    attention_masks = attention_masks[:, None, :].expand(-1, c, -1)
    return seqs_inputs, attention_masks, targets, mask


def train(args: argparse.Namespace):
    setup(args)
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()
    pad_token = tokenizer.token_to_id('[PAD]')
    model = get_model(args, vocab_size=vocab_size, mask=False)
    optimizer, scheduler = learners(model, args)
    tokens_seen = 0
    data_train, data_val, data_test = enwik8(args.data)
    pad_token = tokenizer.token_to_id('[PAD]')
    if args.watch_model:
        wandb.watch(model)
    # We want the mask token index to not be a token in the actual data.
    n_validated = 0
    data_train, data_test = (data_train, data_val)
    batch_loss = 0.
    if args.micro_batch_size is not None:
        num_micro_batches = args.batch_size // args.micro_batch_size
    else:
        num_micro_batches = 1
    mb_size = args.batch_size if args.micro_batch_size is None else args.micro_batch_size
    for i in range(args.num_batches * num_micro_batches):
        model.train(True)
        source, attn_masks, target, mask = sample_batch(data_train,
                                                        tokenizer,
                                                        length=args.context,
                                                        batch_size=mb_size,
                                                        min_length=args.context // 2)
        if cuda:
            source, attn_masks, target, mask = source.cuda(), attn_masks.cuda(), target.cuda(), mask.cuda()

        tokens_seen += (source != pad_token).sum()
        logits = model(source, attn_masks)

        loss = F.cross_entropy(logits[mask].reshape(-1, vocab_size), target[mask].reshape(-1), reduction='mean')
        batch_loss += loss.item()
        loss.backward()
        if i % args.log_every == 0:
            bloss = batch_loss / args.log_every
            wandb.log({
                'loss': bloss,
                'lr': scheduler.get_last_lr()[0],
                'tokens': tokens_seen,
            }, commit=False, step=i)
            batch_loss = 0.

        if i % num_micro_batches == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i % args.validation_every == 0 and i > 0:
            model.train(False)
            source, attn_masks, target, mask = sample_batch(data_test,
                                                            tokenizer,
                                                            length=args.context,
                                                            batch_size=mb_size,
                                                            min_length=args.context // 2)
            if cuda:
                source, attn_masks, target, mask = source.cuda(), attn_masks.cuda(), target.cuda(), mask.cuda()
            logits = model(source, attn_masks)
            loss = F.cross_entropy(logits[mask].reshape(-1, vocab_size), target[mask].reshape(-1), reduction='mean')
            to_log = {'validation_loss': loss.item()}
            print('wandblog', to_log)
            wandb.log(to_log, step=i // args.validation_every, commit=False)
            n_validated += 1
            if args.save_dir is None:
                continue
            if args.plot_attention:
                _, (ms, ss, _) = model.forward_for_plot(source)
                # Iterate through the layers of the model, looking at first item in the batch
                for layer, (m, s) in enumerate(zip(ms, ss)):
                    m = m[0]
                    s = s[0]
                    context = m.size(0)
                    m = m.view(-1, 2)
                    s = s.view(-1)
                    attention_viz(m, s, (context, context),
                                  save_file=f'{args.save_dir}/{n_validated // args.save_every}_attention_{layer}.pdf')
            if n_validated % args.save_every == 0:
                f_name = f'{args.save_dir}/' if args.save_last_only else f'{args.save_dir}/checkpoint_{n_validated // args.save_every}_'
                torch.save(model.state_dict(), f_name + 'model.pt')
                torch.save(optimizer.state_dict(), f_name + 'optimizer.pt')
                torch.save(scheduler.state_dict(), f_name + 'scheduler.pt')


def ttos(t: torch.Tensor, tokenizer: tokenizers.Tokenizer = None) -> str:
    if tokenizer is None:
        return ''.join(map(chr, t))
    return tokenizer.decode(t)


def interact(args):
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()
    model = get_model(args, vocab_size=vocab_size, mask=False)
    data_train, data_val, data_test = enwik8(args.data)
    source, attn_masks, target, mask = sample_batch(data_train,
                                                    tokenizer=tokenizer,
                                                    length=args.context,
                                                    batch_size=args.batch_size,
                                                    min_length=args.context // 2)
    if cuda:
        source, target, mask = source.cuda(), target.cuda(), mask.cuda()

    logits = model(source)
    output = F.log_softmax(logits, dim=-1)
    preds = torch.argmax(output, dim=-1)
    breakpoint()
    print('\n'.join(map(lambda t: ttos(t, tokenizer), [target[0], source[0], preds[0]])))


def find_latest_model(path: str) -> Optional[str]:
    """Finds the latest model saved in path"""
    files = os.listdir(path)
    max_checkpoint = -1
    for f in files:
        match = re.match('.*/?checkpoint_(\d+)_model.pt', f)
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
        save_dir = Path(new_args.save_dir).absolute()
        match = re.match(r'(.*)(\d+)$', save_dir.parts[-1])
        if match is not None:
            next_i = int(match[2]) + 1
            next_path = f'{match[1]}{next_i}'
        else:
            next_i = 2
            next_path = f'{save_dir}{next_i}'
        new_args.__dict__.update(save_dir=next_path)
    if latest_model_checkpoint is not None:
        new_args.__dict__.update(load_model=latest_model_checkpoint)
    new_args.__dict__.update(save_last_only=args.save_last_only)
    return new_args


def main():
    args = parse_args()
    if args.resume_run is not None:
        args = get_resume_args(args)
    print(args)
    init_wandb(args)
    try:
        if args.interact:
            interact(args)
        else:
            train(args)
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
