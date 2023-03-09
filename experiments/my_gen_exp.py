import argparse
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import gzip

import wandb

from transformer_models import GeneratingTransformer

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
        args.num_indices,
        args.n_heads,
        nadditional=args.nadditional,
        gadditional=args.gadditional,
        attention_type=args.attention_type,
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


def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.
    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target


def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.
    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.
    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose:  # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0)  # Append the sampled token to the sequence

    print()
    return seed


def init_wandb(args):
    wandb.init(
        project='sparse-gtransformer',
        config={
            'context': args.context,
            'lr': args.learning_rate,
            'embedding': args.embedding,
            'depth': args.depth,
            'k': args.num_indices,
        }
    )


def train(args: argparse.Namespace):
    model = get_model(args)
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda i: 1.0 if i < 3500 else 0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize, gamma=0.5)
    instances_seen = 0
    data_train, data_val, data_test = enwik8(args.data)
    data_train, data_test = (data_train, data_val)
    for i in range(args.num_batches):
        model.train(True)
        optimizer.zero_grad()
        source, target = sample_batch(data_train, length=args.context, batch_size=args.batch_size)
        if cuda:
            source, target = source.cuda(), target.cuda()
        instances_seen += source.size(0)
        output = model(source)
        loss = torch.nn.functional.nll_loss(output.transpose(2, 1), target, reduction='mean')
        to_log = {'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}
        print('wandblog', to_log)
        wandb.log(to_log)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
        optimizer.step()
        scheduler.step()

        if i % args.validation_every == 0 and i > 0:
            model.train(False)
            source, target = sample_batch(data_test, length=args.context, batch_size=args.batch_size)
            if cuda:
                source, target = source.cuda(), target.cuda()
            instances_seen += source.size(0)
            output = model(source)
            loss = torch.nn.functional.nll_loss(output.transpose(2, 1), target, reduction='mean')
            to_log = {'validation_loss': loss.item()}
            print('wandblog', to_log)
            wandb.log(to_log)


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
                        default=5000, type=int)
    parser.add_argument('-S', '--lr-stepsize',
                        dest='lr_stepsize',
                        default=10, type=int)
    parser.add_argument('-n', '--neighbor_sample',
                        dest='nadditional', default=2, type=int)
    parser.add_argument('-g', '--global_sample',
                        dest='gadditional', default=2, type=int)
    parser.add_argument('-V', '--validation-every', default=200,
                        dest='validation_every', type=int)
    parser.add_argument('-A', '--attention-type', choices=['dense', 'sparse'],
                        dest='attention_type', default='dense', type=str)
    parser.add_argument('-L', '--clipping-value', type=float,
                        dest='clipping_value', default=1.0)
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
