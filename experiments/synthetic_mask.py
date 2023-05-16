"""
This file will hold experiments around making synthetic masks for the MLM task by replacing
tokens with the [MASK] token but then placing the target outside the receptive field for a
normal convolutional model. This will allow us to see if the model can learn to use the
adaptive receptive field to learn to predict the target token.
"""
import os
import git
import torch
import torch.nn.functional as F
import wandb

from mlm_utils import ttos
from experiment_utils import (
    cuda,
    get_tokenizer,
    get_model,
    parse_args,
    setup,
    learners,
    save_model,
)
from mlm_components import enwik8


def init_wandb(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    wandb.init(
        project=f'sparse-transformer-synth',
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
            'random_data': args.rand_data,
            'random_seed': args.seed,
        }
    )


def random_sample_data(batch_size, seq_len, offset=70):
    seqs_inputs = torch.randint(size=(batch_size, seq_len), low=100, high=32000)
    attention_masks = torch.ones_like(seqs_inputs)
    mask_token = 4
    mask = torch.ones((batch_size, seq_len))
    mask[:, 45:105] = 0
    mask = mask.bool()
    targets = seqs_inputs.detach().clone()
    # Modify the input so that the masked token positions are filled with [MASK] tokens
    # and the token at position mask + offset is the target token.
    for b, m_i in (~mask).nonzero():
        seqs_inputs[b] = apply_offset_mask(seqs_inputs[b], m_i, mask_token, offset)
    # Expand the attention mask to a symmetric matrix
    attention_masks = attention_masks[:, None, :].expand(-1, seq_len, -1)
    if cuda:
        seqs_inputs = seqs_inputs.cuda()
        attention_masks = attention_masks.cuda()
        targets = targets.cuda()
        mask = mask.cuda()
    return seqs_inputs, attention_masks, targets, mask


def simple_sample_data(data, tokenizer, batch_size, seq_len, offset=70):
    """
    This function takes the whole dataset as a sequence in data and samples a batch of
    sequences of length seq_len. The sequences are sampled from the dataset randomly
    and the masked token is also sampled randomly within each sequence. The value of the masked
    token is then placed +offset from the masked token in the sequence. Therefore, a convolutional
    model will not be able to see the actual target while the adaptive receptive field model will
    be able to see the target.
    """
    mask_token = tokenizer.token_to_id('[MASK]')
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - seq_len * 4)
    strs = [ttos(data[start:start + seq_len * 4]) for start in starts]
    encoded_ids = []
    attention_masks = []
    for s in strs:
        encoded = tokenizer.encode(s)
        encoded_ids.append(encoded.ids[:seq_len])
        attention_mask = [1] * seq_len
        attention_masks.append(attention_mask)
    seqs_inputs = torch.tensor(encoded_ids)
    attention_masks = torch.tensor(attention_masks).bool()
    mask = torch.ones(size=(batch_size,)).long() * 45
    targets = seqs_inputs.detach().clone()
    # Modify the input so that the masked token positions are filled with [MASK] tokens
    # and the token at position mask + offset is the target token.
    for b, m_i in enumerate(mask):
        for j in range(10):
            seqs_inputs[b] = apply_offset_mask(seqs_inputs[b], m_i + j, mask_token, offset)
    # Expand the attention mask to a symmetric matrix
    attention_masks = attention_masks[:, None, :].expand(-1, seq_len, -1)
    if cuda:
        seqs_inputs = seqs_inputs.cuda()
        attention_masks = attention_masks.cuda()
        targets = targets.cuda()
        mask = mask.cuda()
    return seqs_inputs, attention_masks, targets, mask


def apply_offset_mask(seq_input, i, mask_token, offset):
    """
    This function replaces seq_input[i] with the mask_token and replaces
    seq_input[i+offset] with the target token.
    """
    target_token = seq_input[i].item()
    seq_input[i] = mask_token
    new_pos = (i + offset) % seq_input.size(0)
    seq_input[new_pos] = target_token
    return seq_input


def _train(args):
    setup(args)
    tokenizer = get_tokenizer(args)
    data_train, data_val, data_test = enwik8(args.data)
    model = get_model(args, tokenizer.get_vocab_size())
    optimizer, scheduler = learners(model, args)
    tokens_seen = 0
    if args.watch_model:
        wandb.watch(model)
    for i in range(args.num_batches):
        model.train()
        optimizer.zero_grad()
        if args.rand_data:
            seqs_inputs, attention_masks, targets, mask = random_sample_data(args.batch_size, args.context)
        else:
            seqs_inputs, attention_masks, targets, mask = simple_sample_data(data_train, tokenizer, args.batch_size,
                                                                             args.context)
        logits = model(seqs_inputs, attention_masks)
        num_classes = logits.size(-1)
        flattened_logits = logits.view(-1, num_classes)
        flattened_targets = targets.view(-1)
        flat_mask_idx = (~mask).view(-1).nonzero().view(-1)
        loss = F.cross_entropy(flattened_logits[flat_mask_idx], flattened_targets[flat_mask_idx], reduction='mean')
        if i % args.log_every == 0:
            to_log = {'loss': loss.item(), 'tokens_seen': tokens_seen, 'lr': scheduler.get_last_lr()[0]}
            if 'WANDB_MODE' in os.environ:
                print(to_log)
            wandb.log(to_log, step=i)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
        optimizer.step()
        scheduler.step()
        tokens_seen += seqs_inputs.size(0) * seqs_inputs.size(1)

        if i % args.validation_every == 0:
            model.eval()
            if args.rand_data:
                seqs_inputs, attention_masks, targets, mask = random_sample_data(args.batch_size, args.context)
            else:
                seqs_inputs, attention_masks, targets, mask = simple_sample_data(data_val, tokenizer, args.batch_size,
                                                                                 args.context)
            logits = model(seqs_inputs, attention_masks)
            num_classes = logits.size(-1)
            flattened_logits = logits.view(-1, num_classes)
            flattened_targets = targets.view(-1)
            flat_mask_idx = (~mask).view(-1).nonzero().view(-1)
            loss = F.cross_entropy(flattened_logits[flat_mask_idx], flattened_targets[flat_mask_idx], reduction='mean')

            """
            loss = F.cross_entropy(torch.index_select(logits, 1, mask[0]).view(-1, logits.size(-1)),
                                   torch.index_select(targets, 1, mask[0]).view(-1),
                                   reduction='mean')

            accuracy = (torch.index_select(logits, 1, mask[0]).view(-1, logits.size(-1)).argmax(dim=-1) ==
                        torch.index_select(targets, 1, mask[0]).view(-1)).float().mean()
            """
            to_log = {'val_loss': loss.item()}
            if 'WANDB_MODE' in os.environ:
                print(to_log)
            wandb.log(to_log, step=i)

        if args.save_dir is not None and i % args.save_every == 0:
            save_model(args, model, optimizer, scheduler, i // args.save_every)


def main(args):
    init_wandb(args)
    _train(args)


if __name__ == '__main__':
    main(parse_args())
