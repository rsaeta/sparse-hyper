import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import gzip

from transformer_models import GeneratingTransformer

# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256


def get_model():
    n_blocks = 8
    context = 200
    emb = 4
    vocab_size = NUM_TOKENS
    k = 4
    n_heads = 4
    args = [n_blocks, context, emb, vocab_size, k, n_heads]
    return GeneratingTransformer(*args)


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
    seqs_inputs  = [data[start:start + length] for start in starts]
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

    if verbose: # Print the seed, surrounded by square brackets
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

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    print()
    return seed


def train(context_len: int = 200, batch_size: int = 16, lr: float = 0.01):
    d = f'./data/enwik8'
    model = get_model()
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    instances_seen = 0
    data_train, data_val, data_test = enwik8(d)
    data_train, data_test = (data_train, data_val)
    for i in range(100):
        optimizer.zero_grad()
        source, target = sample_batch(data_train, length=context_len, batch_size=batch_size)
        instances_seen += source.size(0)
        output = model(source)
        loss = torch.nn.functional.nll_loss(output.transpose(2, 1), target, reduction='mean')
        print(loss)
        loss.backward()
        optimizer.step()


def main():
    train()

if __name__ == '__main__':
    main()
