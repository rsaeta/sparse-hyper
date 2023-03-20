import argparse
import json
from argparse import ArgumentParser
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
    return GeneratingTransformer(
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


def train(args: argparse.Namespace):
    model = get_model(args)
    setup(args)
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda i: min(1, (i+1)/args.lr_warmup))
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

        output = model(source)
        loss = torch.nn.functional.nll_loss(output[mask.nonzero(as_tuple=True)],
                                            target[mask.nonzero(as_tuple=True)], reduction='mean')
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
            output = model(source)
            loss = torch.nn.functional.nll_loss(output[mask, :], target[mask], reduction='mean')
            to_log = {'validation_loss': loss.item()}
            print('wandblog', to_log)
            wandb.log(to_log)
            _, (ms, ss, _) = model.forward_for_plot(source)
            # Iterate through the layers of the model
            for layer, m, s in enumerate(zip(ms, ss)):
                context = m.size(1)
                m = m.view(-1, 2)
                s = s.view(-1)
                attention_viz(m, s, (context, context), save_file=f'{args.save_dir}/attention_{n_validated//args.save_every}.pdf')
            if n_validated % args.save_every == 0:
                f_name = f'{args.save_dir}/checkpoint_{n_validated//args.save_every}.pt'
                torch.save(model.state_dict(), f_name)
            n_validated += 1


def parse_args() -> argparse.Namespace:
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
    parser.add_argument('-A', '--attention-type', choices=['dense', 'sparse', 'fixed', 'sparse2d'],
                        dest='attention_type', default='dense', type=str)
    parser.add_argument('-L', '--clipping-value', type=float,
                        dest='clipping_value', default=1.0)
    parser.add_argument('-Y', '--save-every', default=1,
                        type=int, dest='save_every')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='./saved_models')
    parser.add_argument('--watch-model', dest='watch_model', action='store_true')
    options = parser.parse_args()
    print(options)
    return options


def main():
    args = parse_args()
    init_wandb(args)
    train(args)
    wandb.finish()


if __name__ == '__main__':
    main()
