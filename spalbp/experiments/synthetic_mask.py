"""
This file will hold experiments around making synthetic masks for the MLM task by replacing
tokens with the [MASK] token but then placing the target outside the receptive field for a
normal convolutional model. This will allow us to see if the model can learn to use the
adaptive receptive field to learn to predict the target token.
"""
from config import RunConfig
import os
import hydra
from hydra.core.config_store import ConfigStore
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from _context import models
from models import GeneratingTransformer
from utils import (
    cuda,
    device,
    setup,
    learners,
    save_model,
    init_wandb,
    post_process_cfg,
)


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


def _train(cfg: RunConfig):
    model = GeneratingTransformer(cfg.model).to(device)
    setup(cfg)
    optimizer, scheduler = learners(model, cfg)
    tokens_seen = 0
    train_cfg = cfg.experiment.training
    for i in range(train_cfg.num_batches):
        model.train()
        optimizer.zero_grad()

        data_sample = random_sample_data2(train_cfg.batch_size, cfg.experiment.context_size)
        seqs_inputs, attention_masks, targets, mask = data_sample

        logits = model(seqs_inputs, attention_masks)
        num_classes = logits.size(-1)
        flattened_logits = logits.view(-1, num_classes)
        flattened_targets = targets.view(-1)
        flat_mask_idx = (~mask).view(-1).nonzero().view(-1)
        loss = F.cross_entropy(flattened_logits[flat_mask_idx], flattened_targets[flat_mask_idx], reduction='mean')
        if i % train_cfg.log_every == 0:
            to_log = {'loss': loss.item(), 'tokens_seen': tokens_seen, 'lr': scheduler.get_last_lr()[0]}
            if 'WANDB_MODE' in os.environ:
                print(to_log)
            wandb.log(to_log, step=i)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clipping_value)
        optimizer.step()
        scheduler.step()
        tokens_seen += seqs_inputs.size(0) * seqs_inputs.size(1)

        if i % train_cfg.validation_every == 0:
            model.eval()
            eval_sample = random_sample_data2(train_cfg.batch_size, cfg.experiment.context_size)
            seqs_inputs, attention_masks, targets, mask = eval_sample
            logits = model(seqs_inputs, attention_masks)
            num_classes = logits.size(-1)
            flattened_logits = logits.view(-1, num_classes)
            flattened_targets = targets.view(-1)
            flat_mask_idx = (~mask).view(-1).nonzero().view(-1)
            loss = F.cross_entropy(flattened_logits[flat_mask_idx], flattened_targets[flat_mask_idx], reduction='mean')
            to_log = {'val_loss': loss.item()}
            if 'WANDB_MODE' in os.environ:
                print(to_log)
            wandb.log(to_log, step=i)

        if cfg.experiment.save_dir is not None and i % train_cfg.save_every == 0:
            save_model(cfg, model, optimizer, scheduler, i // train_cfg.save_every)


cs = ConfigStore.instance()
cs.store(name='run', node=RunConfig)


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: OmegaConf):
    cfg = post_process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    init_wandb(cfg)
    _train(cfg)


if __name__ == '__main__':
    main()
