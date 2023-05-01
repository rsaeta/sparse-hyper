import argparse
from experiment_utils import parse_args, get_model, cuda, learners
from experiments.mlm_components import enwik8

import torch
import torch.nn.functional as F
import torch.distributions as dist

import wandb


# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256


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
    model = get_model(args, mask=True)
    if cuda:
        model.cuda()
    optimizer, scheduler = learners(model, args)
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
        logits = model(source)
        output = torch.nn.functional.log_softmax(logits, dim=-1)
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
            logits = model(source)
            output = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.nll_loss(output.transpose(2, 1), target, reduction='mean')
            to_log = {'validation_loss': loss.item()}
            print('wandblog', to_log)
            wandb.log(to_log)


def main():
    args = parse_args()
    init_wandb(args)
    train(args)
    wandb.finish()


if __name__ == '__main__':
    main()
