"""
In this file we will explore using the adaptive sparsity for next-bit-prediction
"""
import os

import torch
from torch import nn, Tensor, optim
from spalbp.models.transformer_models import ClassificationTransformer


def get_answers(batch: Tensor, nums: int) -> Tensor:
    indices = bin2dec(batch[:, -nums:], nums)
    return torch.index_select(batch, 1, indices).diag()


def train(model: nn.Module,
          iters: int = 100,
          batch_size: int = 32,
          context_len: int = 200,
          nums_to_decide: int = 6):
    optimizer = optim.Adam(lr=0.1, params=model.parameters())
    for i in range(iters):
        optimizer.zero_grad()
        batch = torch.rand((batch_size, context_len)).round().int()
        y_hat = model(batch)
        y = get_answers(batch, nums_to_decide)[:, None]
        loss = torch.nn.functional.nll_loss(y_hat, y.squeeze(1).long())
        print(loss)
        loss.backward()
        optimizer.step()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def get_model():
    n_blocks = 8
    context = 200
    emb = 4
    vocab_size = 2
    num_classes = 2
    k = 4
    n_heads = 4
    args = [n_blocks, context, emb, vocab_size, num_classes, k, n_heads]
    return ClassificationTransformer(*args)


def load_model():
    fpath = f'{os.path.dirname(os.path.realpath(__file__))}/../adaptive_bits.pt'
    model = get_model()
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model


def main():
    n_blocks = 8
    context = 200
    emb = 1
    vocab_size = 2
    num_classes = 2
    k = 4
    n_heads = 4
    model = get_model()
    train(model, context_len=context)
    torch.save(model.state_dict(), './adaptive_bits.pt')


if __name__ == '__main__':
    main()
