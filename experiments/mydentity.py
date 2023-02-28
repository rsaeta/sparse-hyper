"""
My implementation of the identity experiment which should help me:
1) learn the code base by using it a bit
2) fix all the pep8 formatting issues driving me insane on the other version
"""
import argparse
import sys

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from _context import sparse
from sparse import util
from tqdm import trange
from argparse import ArgumentParser


def get_model(exp_params: argparse.Namespace):
    """
    Just used to get the model given the arguments. Either ReinforceLayer or NASLayer.
    """
    if exp_params.reinforce:
        model = ReinforceLayer(exp_params.size, fix_values=exp_params.fix_values)
    else:
        additional = exp_params.additional \
            if exp_params.additional is not None \
            else int(np.floor(np.log2(exp_params.size)) * exp_params.size)
        model = sparse.NASLayer(
            (exp_params.size,), (exp_params.size,),
            k=exp_params.size,
            gadditional=additional,
            sigma_scale=exp_params.sigma_scale,
            has_bias=False,
            fix_values=exp_params.fix_values,
            min_sigma=exp_params.min_sigma,
            region=(exp_params.rr, exp_params.rr),
            radditional=exp_params.ca,
        )
    if exp_params.cuda:
        model.cuda()
    return model


def _run_subbatched_train_iter(model: nn.Module,
                               optimizer: optim.Optimizer,
                               x: torch.tensor,
                               size: int,
                               subbatch: int) -> float:
    """Runs a training iteration where a subbatch is specified which is useful for running MBGD on large inputs."""
    subbatch_losses = []
    seed = (torch.rand(1) * 100000).long().item()
    for frm in range(0, size, subbatch):
        y = model(x, mrange=(frm, min(frm + subbatch, size)), seed=seed)
        loss = F.mse_loss(y, x)
        loss.backward()
        subbatch_losses.append(loss.item())
    optimizer.step()
    return sum(subbatch_losses)/len(subbatch_losses)


def _run_reinforce_train_iter(model: nn.Module, optimizer: optim.Optimizer, x: torch.tensor):
    """
    This runs a training iteration using the REINFORCE model which approximates gradients using REINFORCE magic on
    log-probability distributions over actions. See https://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/
    for more details.
    """
    y, dists, actions = model(x)
    mse_loss = F.mse_loss(y, x, reduce=False).mean(dim=1)
    reinforce_loss = -dists.log_prob(actions) * -mse_loss[:, None, None].expand_as(actions)
    loss = reinforce_loss.mean()
    loss.backward()
    optimizer.step()


def _run_train_iter(model: nn.Module,
                    optimizer: optim.Optimizer,
                    x: torch.tensor,
                    size: int = None,
                    subbatch: int = None):
    """
    Executes a single iteration of normal training. Because this is the identity experiment, a targets dataset is not
    needed as it is just x itself. It calls the optimizer to step through a single step of GD as well.

    :param model: the model on which to execute the training
    :param optimizer: the optimizer to step through GD
    :param x: the batch of input data on which to train
    :param size: the size of the input batch
    :param subbatch: the size of the subbatch in batch training
    :returns: the loss in this training batch
    """
    optimizer.zero_grad()
    if type(model) == ReinforceLayer:  # Here we have to use REINFORCE (gradient estimation with many iters)
        return _run_reinforce_train_iter(model, optimizer, x)
    else:
        if subbatch is None:
            breakpoint()
            y = model(x)
            loss = F.mse_loss(y, x)
            loss.backward()
            optimizer.step()
            return loss.item()
        return _run_subbatched_train_iter(model, optimizer, x, size, subbatch)


def _run_validation(model: nn.Module, size: int, batch: int):
    model.train(False)
    losses = []
    with torch.no_grad():
        for frm in range(0, 10000, batch):
            to = min(frm+batch, 10000)
            x = torch.randn(to-frm, size)
            if model.is_cuda():
                x = x.cuda()
            x = torch.autograd.Variable(x)
            if type(model) == ReinforceLayer:
                y, _, _ = model(x)
            else:
                y = model(x)
            val_loss = F.mse_loss(y, x)
            losses.append(val_loss.item())
    return torch.tensor(losses)


def run(exp_params: argparse.Namespace):
    """
    Responsible for using the arguments passed to run an experiment
    """
    torch.manual_seed(exp_params.seed)
    iterations = exp_params.iterations if exp_params.iterations is not None else exp_params.size * 3000
    model = get_model(exp_params)
    optimizer = optim.Adam(model.parameters(), lr=exp_params.lr)
    n_dots = iterations // exp_params.dot_every
    results = np.zeros((exp_params.reps, n_dots))  # Hold the validation results during training
    for r in range(exp_params.reps):
        with SummaryWriter(log_dir=f'./runs/identity/{r}') as w:
            print(f'Running repetition {r+1} of {exp_params.reps}')
            util.makedirs(f'./identity/{r}')
            util.makedirs(f'./runs/identity/{r}')
            for i in trange(iterations):  # Training loop
                model.train(True)
                x = torch.randn((exp_params.batch,) + (exp_params.size,))
                if exp_params.cuda:
                    x = x.cuda()
                x = torch.autograd.Variable(x)
                training_loss = _run_train_iter(model, optimizer, x)
                w.add_scalar('identity/train_loss/', training_loss, i*(r+1))
                if i % exp_params.dot_every == 0:  # Run a validation batch to see results
                    val_losses = _run_validation(model, exp_params.size, exp_params.batch)
                    mean_val_loss = val_losses.sum().item() / val_losses.size(0)
                    results[r, i // exp_params.dot_every] = mean_val_loss
                    w.add_scalar('identity/val_loss/', mean_val_loss, (i // exp_params.dot_every) * (r + 1))


class ReinforceLayer(nn.Module):
    """
    Baseline method: use REINFORCE to sample from the continuous index tuples.
    """
    def __init__(self, size,
                 sigma_scale=0.2,
                 fix_values=False,
                 min_sigma=0.0):

        super().__init__()

        self.size = size
        self.sigma_scale = sigma_scale
        self.fix_values = fix_values
        self.min_sigma = min_sigma

        self.pmeans = nn.Parameter(torch.randn(self.size, 2))
        self.psigmas = nn.Parameter(torch.randn(self.size))

        if fix_values:
            self.register_buffer('pvalues', torch.ones(self.size))
        else:
            self.pvalues = nn.Parameter(torch.randn(self.size))

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def hyper(self, x):

        b = x.size(0)
        size = (self.size, self.size)

        # Expand parameters along batch dimension
        means = self.pmeans[None, :, :].expand(b, self.size, 2)
        sigmas = self.psigmas[None, :].expand(b, self.size)
        values = self.pvalues[None, :].expand(b, self.size)

        means, sigmas = sparse.transform_means(means, size), sparse.transform_sigmas(sigmas, size)

        return means, sigmas, values

    def forward(self, x):
        size = (self.size, self.size)

        means, sigmas, values = self.hyper(x)

        dists = torch.distributions.Normal(means, sigmas)
        samples = dists.sample()

        indices = samples.data.round().long()

        # if the sampling puts the indices out of bounds, we just clip to the min and max values
        indices[indices < 0] = 0

        rngt = torch.tensor(data=size, device='cuda' if self.is_cuda() else 'cpu')
        maxes = rngt[None, None, :].expand_as(means) - 1
        indices[indices > maxes] = maxes[indices > maxes]

        y = sparse.contract(indices, values, size, x)

        return y, dists, samples


def parse_args():
    """
    Definitely stole this from the original experiment, but doesn't matter. Parse some args not that hard.
    """
    # Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="Size (nr of dimensions) of the input.",
                        default=10000, type=int)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size (nr of dimensions) of the input.",
                        default=16, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="additional",
                        help="Number of global additional points sampled ",
                        default=4, type=int)

    parser.add_argument("-R", "--rrange",
                        dest="rr",
                        help="Size of the sampling region around the index tuple.",
                        default=4, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="ca",
                        help="Number of points to sample from the sampling region.",
                        default=4, type=int)

    parser.add_argument("-C", "--sub-batch",
                        dest="subbatch",
                        help="Size for updating in multiple forward/backward passes.",
                        default=None, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-F", "--fix_values", dest="fix_values",
                        help="Whether to fix the values to 1.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.005, type=float)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale",
                        default=0.1, type=float)

    parser.add_argument("-M", "--min_sigma",
                        dest="min_sigma",
                        help="Minimum variance for the components.",
                        default=0.0, type=float)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Plot every x iterations",
                        default=1000, type=int)

    parser.add_argument("-d", "--dot-every",
                        dest="dot_every",
                        help="A dot in the graph for every x iterations",
                        default=1000, type=int)

    parser.add_argument("--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=1, type=int)

    parser.add_argument("--seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    parser.add_argument("-B", "--use-reinforce", dest="reinforce",
                        help="Use the reinforce baseline instead of the backprop approach.",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS ', options)
    return options


if __name__ == '__main__':
    args = parse_args()
    run(args)
