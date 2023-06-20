import torch
from utils import cuda


def random_sample_simple(batch_size, seq_len, mask_i=0, target_i=-1):
    seqs_inputs = torch.randint(size=(batch_size, seq_len), low=100, high=32000)
    attention_masks = torch.ones_like(seqs_inputs)
    mask = torch.ones((batch_size, seq_len))
    mask[:, mask_i] = 0
    mask_token = 4
    targets = seqs_inputs.detach().clone()
    seqs_inputs[:, mask_i] = mask_token
    seqs_inputs[:, target_i] = targets[:, mask_i]
    # Expand the attention mask to a symmetric matrix
    attention_masks = attention_masks[:, None, :].expand(-1, seq_len, -1)
    mask = mask.bool()
    if cuda:
        seqs_inputs = seqs_inputs.cuda()
        attention_masks = attention_masks.cuda()
        targets = targets.cuda()
        mask = mask.cuda()
    return seqs_inputs, attention_masks, targets, mask


def random_sample_data2(batch_size, seq_len, offset=70):
    seqs_inputs = torch.randint(size=(batch_size, seq_len), low=100, high=32000)
    attention_masks = torch.ones_like(seqs_inputs)
    mask_token = 4
    mask = torch.rand((batch_size, seq_len)) > 0.05
    # mask = torch.ones((batch_size, seq_len))
    # mask[:, 45:55] = 0
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