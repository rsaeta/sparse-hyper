import argparse
import json
from exp_args import parse_args
from functools import partial
import os

import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import gzip

import wandb

from transformer_models import GeneratingTransformer
from plot_utils import attention_viz

# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256

cuda = torch.cuda.is_available()
print(f'Cuda is: {cuda}')


def get_model(args: argparse.Namespace) -> GeneratingTransformer:
    model = GeneratingTransformer(
        args.depth,
        args.context,
        args.embedding,
        NUM_TOKENS,
        k=args.num_indices,
        heads=args.n_heads,
        nadditional=args.nadditional,
        gadditional=args.gadditional,
        attention_type=args.attention_type,
        mask=False,
    )
    if args.load_model is not None:
        state_dict = torch.load(args.load_model, map_location=torch.device('cuda') \
                                if cuda else torch.device('cpu'))
        model.load_state_dict(state_dict)
    if cuda:
        model = model.cuda()
    return model


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()


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
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)


def sample_batch(data, length, batch_size, mask_token, mask_p=0.15):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model
    while also randomly masking a single entry in the sequence to the mask_token provided.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :param mask_token: The token that represents the mask to be predicted by the model
    :param mask_p: The probability of masking a token
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs = torch.cat([data[None, start:start + length] for start in starts]).long()
    mask = torch.rand(seqs_inputs.size()) < mask_p
    targets = seqs_inputs.clone()
    seqs_inputs.masked_fill_(mask, mask_token)

    return seqs_inputs, targets, mask


def init_wandb(args):
    wandb.init(
        project='sparse-masked-transformer',
        config={
            'context': args.context,
            'lr': args.learning_rate,
            'embedding': args.embedding,
            'depth': args.depth,
            'k': args.num_indices,
        }
    )


def setup(args: argparse.Namespace):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

def lr(args, i):
    if i < args.lr_warmup:
        return (i+1)/args.lr_warmup
    else:
        return 1 - i/args.num_batches


def train(args: argparse.Namespace):
    model = get_model(args)
    setup(args)
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  partial(lr, args))
    instances_seen = 0
    data_train, data_val, data_test = enwik8(args.data)
    if args.watch_model:
        wandb.watch(model)
    # We want the mask token index to not be a token in the actual data.
    mask_token_index = torch.cat([data_train, data_val, data_test], dim=0).max().item() + 1
    n_validated = 0
    data_train, data_test = (data_train, data_val)
    for i in range(args.num_batches):
        model.train(True)
        optimizer.zero_grad()
        source, target, mask = sample_batch(data_train,
                                            length=args.context,
                                            batch_size=args.batch_size,
                                            mask_token=mask_token_index)
        if cuda:
            source, target, mask = source.cuda(), target.cuda(), mask.cuda()
        instances_seen += source.size(0)

        logits = model(source)
        output = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.nll_loss(output[mask],
                                            target[mask],
                                            reduction='mean')
        to_log = {'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}
        print('wandblog', to_log)
        wandb.log(to_log)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
        optimizer.step()
        scheduler.step()

        if i % args.validation_every == 0 and i > 0:
            model.train(False)
            source, target, mask = sample_batch(data_test,
                                                length=args.context,
                                                batch_size=args.batch_size,
                                                mask_token=mask_token_index)
            if cuda:
                source, target, mask = source.cuda(), target.cuda(), mask.cuda()
            instances_seen += source.size(0)
            logits = model(source)
            output = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.nll_loss(output[mask],
                                                target[mask],
                                                reduction='mean')
            to_log = {'validation_loss': loss.item()}
            print('wandblog', to_log)
            wandb.log(to_log)
            if args.plot_attention:
                _, (ms, ss, _) = model.forward_for_plot(source)
                # Iterate through the layers of the model, looking at first item in the batch
                for layer, (m, s) in enumerate(zip(ms, ss)):
                    m = m[0]
                    s = s[0]
                    context = m.size(0)
                    m = m.view(-1, 2)
                    s = s.view(-1)
                    attention_viz(m, s, (context, context), save_file=f'{args.save_dir}/{n_validated//args.save_every}_attention_{layer}.pdf')
            if n_validated % args.save_every == 0:
                f_name = f'{args.save_dir}/checkpoint_{n_validated//args.save_every}.pt'
                torch.save(model.state_dict(), f_name)
            n_validated += 1


def ttos(t: torch.Tensor) -> str:
    return ''.join(map(chr, t))


def interact(args):
    model = get_model(args)
    data_train, data_val, data_test = enwik8(args.data)
    mask_token_index = torch.cat([data_train, data_val, data_test], dim=0).max().item() + 1
    source, target, mask = sample_batch(data_train,
                                        length=args.context,
                                        batch_size=args.batch_size,
                                        mask_token=mask_token_index)
    if cuda:
        source, target, mask = source.cuda(), target.cuda(), mask.cuda()

    logits = model(source)
    output = torch.nn.functional.log_softmax(logits, dim=-1)
    preds = torch.argmax(output, dim=-1)
    breakpoint()
    print('\n'.join(map(ttos, [target[0], source[0], preds[0]])))


def main():
    args = parse_args()
    if args.interact:
        interact(args)
        return
    init_wandb(args)
    train(args)
    wandb.finish()


if __name__ == '__main__':
    main()
