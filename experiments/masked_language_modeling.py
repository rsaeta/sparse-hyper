import os
import argparse
import git

from experiment_utils import (
    parse_args,
    get_model,
    learners,
    setup,
    get_tokenizer,
    cuda, 
    get_resume_args,
    save_model,
)
from mlm_components import enwik8

import torch
import torch.nn.functional as F

import wandb

from mlm_utils import sample_batch, ttos
from plot_utils import attention_viz


def init_wandb(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    proj_suffix = 'prod' if args.production else 'dev'
    wandb.init(
        project=f'sparse-masked-transformer-{proj_suffix}',
        config={
            'context': args.context,
            'lr': args.learning_rate,
            'embedding': args.embedding,
            'depth': args.depth,
            'k': args.num_indices,
            'attention': args.attention_type,
            'gitsha': git.Repo(dir_path, search_parent_directories=True).head.object.hexsha,
            'model_type': args.model_type,
            'dir_path': dir_path,
        }
    )


def train(args: argparse.Namespace):
    setup(args)
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()
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
        model.train()
        source, attn_masks, target, mask = sample_batch(data_train,
                                                        tokenizer,
                                                        length=args.context,
                                                        batch_size=mb_size,
                                                        min_length=args.context // 3)
        
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

        # Run backpropagation every num_micro_batches
        if i % num_micro_batches == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Run validation every validation_every batches
        if i % args.validation_every == 0 and i > 0:
            model.eval()
            source, attn_masks, target, mask = sample_batch(data_test,
                                                            tokenizer,
                                                            length=args.context,
                                                            batch_size=mb_size,
                                                            min_length=args.context // 3)
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
                save_model(args, model, optimizer, scheduler, n_validated // args.save_every)


def interact(args):
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()
    model = get_model(args, vocab_size=vocab_size, mask=False)
    data_train, data_val, data_test = enwik8(args.data)
    source, attn_masks, target, mask = sample_batch(data_train,
                                                    tokenizer=tokenizer,
                                                    length=args.context,
                                                    batch_size=args.micro_batch_size,
                                                    min_length=args.context // 2 - 5)
    if cuda:
        source, attn_masks, target, mask = source.cuda(), attn_masks.cuda(), target.cuda(), mask.cuda()

    logits = model(source, attn_masks)
    output = F.log_softmax(logits, dim=-1)
    preds = torch.argmax(output, dim=-1)
    breakpoint()
    print('\n'.join(map(lambda t: ttos(t, tokenizer), [target[0], source[0], preds[0]])))


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


def derp():
    args = argparse.Namespace()
    args.__dict__.update(resume_run='/home/rsaeta/sparse-hyper/dabirds3', save_last_only=True, save_dir=None)
    new_args = get_resume_args(args)
    print(new_args)


if __name__ == '__main__':
    main()
