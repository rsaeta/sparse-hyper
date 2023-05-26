import argparse
import os
import json

from _context import sparse
from spalbp.utils import (
    get_model,
    get_tokenizer,
    find_latest_model,
)

from spalbp.synthetic_mask import random_sample_data2


def load_model(path_dir):
    config_fname = os.path.join(path_dir, 'config.json')
    new_args = argparse.Namespace()
    with open(config_fname, 'r') as f:
        new_args.__dict__.update(json.load(f))
    latest_model = find_latest_model(path_dir)
    if latest_model is not None:
        new_args.__dict__.update(load_model=latest_model)
    tokenizer = get_tokenizer(new_args)
    vocab_size = tokenizer.get_vocab_size()
    model = get_model(new_args, vocab_size=vocab_size, mask=False)
    # inpt, attn_masks, targets, mask = simple_sample_data(data, tokenizer, 1, new_args.context)
    inpt, attn_masks, targets, mask = random_sample_data2(4, new_args.context)
    emb = model.embed(inpt)
    means, sigmas, values = model.t_blocks[0].attend.hyper(emb)
    train_indices = sparse.ngenerate(means, new_args.gadditional, new_args.nadditional, rng=(new_args.context,),
                                     relative_range=(3,), cuda=True)  # (B, C, P, 1)
    eval_indices = sparse.ngenerate(means, 0, 0, rng=(new_args.context,),
                                    relative_range=(3,), cuda=True)  # (B, C, P, 1)

    breakpoint()


if __name__ == '__main__':
    load_model('synth-rand-sparse-1-depth')
