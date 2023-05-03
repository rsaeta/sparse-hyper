import tokenizers
import torch

cuda = torch.cuda.is_available()


def sample_batch(data, tokenizer, length, batch_size, min_length, mask_p=0.15):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model
    while also randomly masking a single entry in the sequence to the mask_token provided.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :param tokenizer: The tokenizer that is used to parse the texts
    :param min_length: the min length of sequences to train on variable-length documents
    :param mask_p: The probability of masking a token
    :return: A tuple (input, attn_mask, target, masking_masks) where
        input: The input to the model. A tensor of shape (batch_size, length)
        attn_mask: The attention mask for the input where tokens that are padding are not attended to
        target: The target for the model. A tensor of shape (batch_size, length)
        masking_masks: A tensor of shape (batch_size, length) where each entry is True if the corresponding entry in
            input was masked and False otherwise
    """
    mask_token = tokenizer.token_to_id('[MASK]')
    pad_token = tokenizer.token_to_id('[PAD]')
    sep_token = tokenizer.token_to_id('[SEP]')
    byte_len = 10 * length

    max_frag_len = (length // 2) - 1

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size, 2), low=0, high=data.size(0) - byte_len)
    lens = torch.randint(size=(batch_size, 2), low=min_length, high=max_frag_len)

    # Slice out the input sequences
    strs = [[ttos(data[startA:startA + byte_len]), ttos(data[startB:startB + byte_len])] for (startA, startB) in starts]

    attention_masks = []
    encoded_ids = []
    for (s1, s2), (l1, l2) in zip(strs, lens):
        encoded1 = tokenizer.encode(s1)
        encoded2 = tokenizer.encode(s2)
        ec_sb1 = encoded1.ids[0:l1]
        ec_sb2 = encoded2.ids[1:l2+1]  # skip the first token, which is the start token
        num_pad1 = max_frag_len - len(ec_sb1)
        num_pad2 = max_frag_len - len(ec_sb2)
        encoded = [*ec_sb1,
                   *([pad_token] * num_pad1),
                   sep_token,
                   *ec_sb2,
                   *([pad_token] * num_pad2)]
        encoded = [*encoded, *([pad_token] * (length - len(encoded)))]
        encoded_ids.append(encoded)
        attention_mask = [*([1] * l1), *([0] * num_pad1), 1, *([1] * l2), *([0] * num_pad2)]
        attention_mask = [*attention_mask, *([0] * (length - len(attention_mask)))]
        attention_masks.append(attention_mask)
    seqs_inputs = torch.tensor(encoded_ids)
    attention_masks = torch.tensor(attention_masks).bool()
    mask = torch.logical_and((torch.rand(seqs_inputs.size()) < mask_p), attention_masks)
    mask = torch.logical_and(mask, ~(seqs_inputs == sep_token))
    targets = seqs_inputs.detach().clone()
    seqs_inputs.masked_fill_(mask, mask_token)
    c = attention_masks.size(-1)
    attention_masks = attention_masks[:, None, :].expand(-1, c, -1)
    if cuda:
        seqs_inputs = seqs_inputs.cuda()
        attention_masks = attention_masks.cuda()
        targets = targets.cuda()
        mask = mask.cuda()
    return seqs_inputs, attention_masks, targets, mask


def ttos(t: torch.Tensor, tokenizer: tokenizers.Tokenizer = None) -> str:
    if tokenizer is None:
        return ''.join(map(chr, t))
    return tokenizer.decode(t)
