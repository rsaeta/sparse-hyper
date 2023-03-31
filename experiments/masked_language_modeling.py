import argparse

import tokenizers

from experiment_utils import (
    parse_args,
    get_model,
    learners,
    cuda,
    setup,
    enwik8,
    get_tokenizer,
)

import torch
import torch.nn.functional as F

import wandb

from plot_utils import attention_viz


def sample_batch(data, tokenizer, length, batch_size, mask_p=0.15):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model
    while also randomly masking a single entry in the sequence to the mask_token provided.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :param tokenizer: The tokenizer that is used to parse the texts
    :param mask_p: The probability of masking a token
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """
    mask_token = tokenizer.token_to_id('[MASK]')
    byte_len = 50*length

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - byte_len)

    # Slice out the input sequences
    strs = [ttos(data[start:start + byte_len]) for start in starts]
    encoded = [torch.tensor(tokenizer.encode(s).ids)[None, :length].long() for s in strs]
    seqs_inputs = torch.cat(encoded)
    mask = torch.rand(seqs_inputs.size()) < mask_p
    targets = seqs_inputs.detach().clone()
    seqs_inputs.masked_fill_(mask, mask_token)
    if cuda:
        seqs_inputs = seqs_inputs.cuda()

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


def train(args: argparse.Namespace):
    setup(args)
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()
    model = get_model(args, vocab_size=vocab_size, mask=False)
    optimizer, scheduler = learners(model, args)
    instances_seen = 0
    data_train, data_val, data_test = enwik8(args.data)

    if args.watch_model:
        wandb.watch(model)
    # We want the mask token index to not be a token in the actual data.
    n_validated = 0
    data_train, data_test = (data_train, data_val)
    batch_loss = 0.
    for i in range(args.num_batches):
        model.train(True)
        optimizer.zero_grad()
        source, target, mask = sample_batch(data_train,
                                            tokenizer,
                                            length=args.context,
                                            batch_size=args.batch_size)
        if cuda:
            source, target, mask = source.cuda(), target.cuda(), mask.cuda()
        instances_seen += source.size(0)

        logits = model(source)
    
        loss = F.cross_entropy(logits[mask].reshape(-1, vocab_size), target[mask].reshape(-1), reduction='mean')
        batch_loss += loss.item()
        if i % args.log_every == 0:
            bloss = batch_loss / args.log_every
            to_log = {'loss': bloss, 'lr': scheduler.get_last_lr()[0]}
            print('wandblog', to_log)
            wandb.log(to_log)
            batch_loss = 0.

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
        optimizer.step()
        scheduler.step()

        if i % args.validation_every == 0 and i > 0:
            model.train(False)
            source, target, mask = sample_batch(data_test,
                                                tokenizer,
                                                length=args.context,
                                                batch_size=args.batch_size)
            if cuda:
                source, target, mask = source.cuda(), target.cuda(), mask.cuda()
            instances_seen += source.size(0)
            logits = model(source)
            loss = F.cross_entropy(logits[mask].reshape(-1, vocab_size), target[mask].reshape(-1), reduction='mean')
            to_log = {'validation_loss': loss.item()}
            print('wandblog', to_log)
            wandb.log(to_log)
            n_validated += 1
            if args.save_dir is None:
                continue
            if args.plot_attention:
                _, (ms, ss, _) = model.forward_for_plot(source)
                # Iterate through the layers of the model, looking at first item in the batch
                for layer, (m, s) in enumerate(zip(ms, ss)):
                    m = m[0]
                    s = s[0]
                    context = m.size(0)
                    m = m.view(-1, 2)
                    s = s.view(-1)
                    attention_viz(m, s, (context, context), save_file=f'{args.save_dir}/{n_validated//args.save_every}_attention_{layer}.pdf')
            if n_validated % args.save_every == 0:
                f_name = f'{args.save_dir}/checkpoint_{n_validated//args.save_every}_'
                torch.save(model.state_dict(), f_name + 'model.pt')
                torch.save(optimizer.state_dict(), f_name + 'optimizer.pt')
                torch.save(scheduler.state_dict(), f_name + 'scheduler.pt')


def ttos(t: torch.Tensor, tokenizer: tokenizers.Tokenizer = None) -> str:
    if tokenizer is None:
        return ''.join(map(chr, t))
    return tokenizer.decode(t)


def interact(args):
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()
    model = get_model(args, vocab_size=vocab_size, mask=False)
    data_train, data_val, data_test = enwik8(args.data)
    source, target, mask = sample_batch(data_train,
                                        tokenizer=tokenizer,
                                        length=args.context,
                                        batch_size=args.batch_size)
    if cuda:
        source, target, mask = source.cuda(), target.cuda(), mask.cuda()

    logits = model(source)
    output = F.log_softmax(logits, dim=-1)
    preds = torch.argmax(output, dim=-1)
    breakpoint()
    print('\n'.join(map(lambda t: ttos(t, tokenizer), [target[0], source[0], preds[0]])))


def main():
    args = parse_args()
    if args.interact:
        interact(args)
        return
    init_wandb(args)
    train(args)
    wandb.finish()


if __name__ == '__main__':
    main()
